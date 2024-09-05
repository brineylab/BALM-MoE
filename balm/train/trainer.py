#!/usr/bin/python
# filename: trainer.py

#
# Copyright (c) 2024 Bryan Briney
# License: GNU General Public License, version 3.0 (http://opensource.org/licenses/gpl-3-0/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


import json
import math
import os
import random
import warnings
from datetime import datetime
from functools import cached_property
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import DataCollator, Dataset
from ..modules import MaskedLMOutput

# from ..tokenizer import TokenizerBase
# from .training_arguments import TrainingArguments
from .training_utils import EvalOutput, EvalPrediction, get_scheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        # args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        eval_collator: Optional[DataCollator] = None,
        per_device_train_batch_size: Optional[int] = 1,
        per_device_eval_batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        logging_steps: Optional[int] = None,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = 1,
        warmup_steps: Optional[int] = 0,
        weight_decay: Optional[float] = 0.01,
        learning_rate: Optional[float] = 4e-4,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        deepspeed: bool = False,
        deepspeed_config: Optional[str] = None,
        use_cpu: bool = False,
        seed: Optional[int] = 42,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        output_dir: Optional[str] = None,
        logging_dir: Optional[str] = None,
        use_wandb: bool = False,
        run_name: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: str = "brineylab",
        # callbacks: Optional[List[TrainerCallback]] = None,
    ):
        warnings.filterwarnings(
            "ignore",
            module="torch.nn.parallel*",
            message=(
                "Was asked to gather along dimension 0, but all"
                " input tensors were scalars; will instead unsqueeze"
                " and return a vector."
            ),
        )
        # if args is None:
        #     output_dir = "tmp_trainer"
        #     args = TrainingArguments(output_dir=output_dir)
        # self.args = args

        # seed gets set before anything else happens
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

        # model and model config
        self.model = model
        if hasattr(model, "config"):
            self.model_config = model.config
        else:
            self.model_config = None

        # data
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_collator = (
            eval_collator if eval_collator is not None else data_collator
        )

        # hyperparameters
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.epochs = epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.deepspeed = deepspeed
        self.deepspeed_config = deepspeed_config
        self.compute_metrics = compute_metrics
        self.use_cpu = use_cpu

        # wandb
        self.use_wandb = use_wandb
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.run_name = (
            run_name
            if run_name is not None
            else f"{self.model.__class__.__name__}_{self.timestamp}"
        )

        # directories
        self.output_dir = (
            os.path.abspath(output_dir) if output_dir is not None else "tmp_trainer"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.logging_dir = (
            os.path.abspath(logging_dir)
            if logging_dir is not None
            else os.path.join(self.output_dir, "log")
        )
        os.makedirs(self.logging_dir, exist_ok=True)
        if self.save_steps is not None:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    @cached_property
    def device(self):
        if torch.cuda.is_available() and not self.use_cpu:
            return torch.device("cuda")
        # elif torch.backends.mps.is_available() and not self.use_cpu:
        #     return torch.device("mps")
        else:
            return torch.device("cpu")

    @cached_property
    def device_count(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1

    @property
    def num_epochs(self):
        if self.max_steps is None:
            return self.epochs
        return math.ceil(
            self.max_steps * self.total_train_batch_size / len(self.train_dataset)
        )

    @property
    def num_train_steps(self):
        if self.max_steps is not None:
            return self.max_steps
        return self.num_epochs * len(self.train_dataset) // self.total_train_batch_size

    @property
    def num_warmup_steps(self):
        if self.warmup_steps < 1:  # warmup ratio if less than 1
            return int(self.num_train_steps * self.warmup_steps)
        return self.warmup_steps

    @property
    def total_train_batch_size(self):
        return self.device_count * self.per_device_train_batch_size

    @property
    def total_eval_batch_size(self):
        if self.eval_dataset is None:
            return 0
        batch_size = self.per_device_eval_batch_size
        if batch_size is None:
            batch_size = self.per_device_train_batch_size
        return batch_size * self.device_count

    @property
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.total_train_batch_size,
            shuffle=True,
            # collate_fn=self.data_collator,
        )

    @property
    def eval_dataloader(self):
        if self.eval_dataset is None:
            return []
        return DataLoader(
            self.eval_dataset,
            batch_size=self.total_eval_batch_size,
            shuffle=False,
            # collate_fn=self.data_collator,
        )

    def train(self):
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.run_name,
                dir=self.logging_dir,
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step", step_sync=True)

        self.model, self.optimizer = self.wrap_model()
        self.model.train()

        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        completed_steps = 0
        pbar = tqdm(total=self.num_train_steps, unit="step", desc="Training")
        for epoch in range(self.num_epochs):
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                collated = self.data_collator(batch)
                inputs = self.place_inputs(collated)
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    labels=inputs.get("labels", None),
                    attention_mask=inputs.get("attention_mask", None),
                    key_padding_mask=inputs.get("key_padding_mask", None),
                )
                if self.device_count > 1:
                    outputs["raw_loss"] = outputs["loss"].clone()
                    outputs["loss"] = outputs["loss"].mean()
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # logging
                completed_steps += 1
                pbar.update(1)
                if completed_steps % self.logging_steps == 0:
                    self.print_train_log(
                        steps=completed_steps,
                        outputs=outputs,
                        lr=self.optimizer.param_groups[0]["lr"],
                        num_train_steps=self.num_train_steps,
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/lr": self.optimizer.param_groups[0]["lr"],
                                "global_step": completed_steps,
                                "train/epoch": epoch,
                            }
                        )

                # eval
                if (
                    self.eval_steps is not None
                    and self.eval_dataset is not None
                    and completed_steps % self.eval_steps == 0
                ):
                    eval_output = self.evaluate(compute_metrics=self.compute_metrics)
                    self.print_eval_log(eval_output, self.num_train_steps)
                    if self.use_wandb:
                        eval_log_dict = {
                            "eval/loss": eval_output.loss,
                            "global_step": completed_steps,
                        }
                        for key, value in eval_output.metrics.items():
                            eval_log_dict[f"eval/{key}"] = value
                        wandb.log(eval_log_dict)

                # checkpoint
                if (
                    self.save_steps is not None
                    and completed_steps % self.save_steps == 0
                    and completed_steps < self.num_train_steps
                ):
                    print("<< SAVING MODEL CHECKPOINT >>")
                    checkpoint_name = f"{self.run_name}_steps={completed_steps}"
                    checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
                    os.makedirs(checkpoint_path, exist_ok=True)
                    model_to_save = self.unwrap_model(self.model)
                    model_to_save.save_pretrained(checkpoint_path)

                # done!
                if completed_steps >= self.num_train_steps:
                    print("<< SAVING FINAL MODEL >>")
                    save_path = os.path.join(self.output_dir, "model")
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = self.unwrap_model(self.model)
                    model_to_save.save_pretrained(save_path)
                    print("\nTraining complete")
                    break
        pbar.close()
        if self.use_wandb:
            wandb.finish()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ) -> EvalOutput:
        if eval_dataset is not None:
            num_eval_steps = len(eval_dataset) // self.total_eval_batch_size
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.total_eval_batch_size,
                shuffle=False,
                # collate_fn=self.data_collator,
            )
        elif self.eval_dataset is not None:
            num_eval_steps = len(self.eval_dataset) // self.total_eval_batch_size
            eval_dataloader = self.eval_dataloader
        else:
            raise ValueError("No evaluation dataset provided")

        self.model.eval()  # Set the model to evaluation mode
        eval_loss = 0.0
        all_logits = []
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Disable gradient calculation
            eval_pbar = tqdm(
                total=num_eval_steps,
                desc="Evaluating: ",
                unit="step",
                leave=False,
            )
            for batch in eval_dataloader:
                collated = self.data_collator(batch)
                inputs = self.place_inputs(collated)
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    labels=inputs["labels"],
                    attention_mask=inputs.get("attention_mask", None),
                    key_padding_mask=inputs.get("key_padding_mask", None),
                )
                tmp_eval_loss = outputs["loss"]
                eval_loss += tmp_eval_loss.mean().item()
                eval_pbar.update(1)

                if "logits" in outputs:
                    all_logits.append(outputs["logits"].detach().cpu())
                    all_preds.append(outputs["logits"].argmax(dim=-1).detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())

        eval_pbar.close()

        all_logits = torch.cat(all_logits)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metric_results = {}
        if compute_metrics is not None:
            metric_results = compute_metrics(
                EvalPrediction(
                    predictions=all_preds,
                    labels=all_labels,
                    logits=all_logits,
                )
            )

        eval_loss = eval_loss / num_eval_steps

        if "accuracy" not in metric_results and self.data_collator.mlm:
            # only compute accuracy if mlm is enabled and it isn't
            # already computed in compute_metrics
            predictions = F.softmax(all_logits, dim=-1).argmax(dim=-1)
            label_mask = all_labels != -100
            correct_predictions = torch.sum((predictions == all_labels) * label_mask)
            metric_results["accuracy"] = correct_predictions.float() / torch.sum(
                label_mask
            )
        if "perplexity" not in metric_results and self.data_collator.mlm:
            # only compute perplexity if mlm is enabled and it isn't
            # already computed in compute_metrics
            logits_flat = all_logits.view(-1, all_logits.size(-1))
            labels_flat = all_labels.view(-1)
            ce_loss = F.cross_entropy(
                logits_flat, labels_flat, ignore_index=-100, reduction="mean"
            )
            perplexity = torch.exp(ce_loss)
            metric_results["perplexity"] = perplexity.item()

        eval_output = EvalOutput(
            loss=eval_loss,
            predictions=outputs["logits"].argmax(dim=-1).cpu().numpy(),
            labels=batch["labels"].cpu().numpy(),
            metrics=metric_results,
            num_samples=len(eval_dataloader),
        )

        return eval_output

    def terminate(self):
        """
        Clean up resources. This will free up GPU memory by removing the
        model, optimizer, and scheduler from memory.
        """
        del self.model
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "scheduler"):
            del self.scheduler
        torch.cuda.empty_cache()

    def save_model(self, save_directory):
        """
        Saves the model to the specified directory.

        Parameters
        ----------
        save_directory : str
            Directory to save the model.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        # torch.save(self.model.state_dict(), os.path.join(save_directory, "model.pt"))
        # if self.model_config is not None:
        #     self.model_config.to_json(os.path.join(save_directory, "config.json"))
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(save_directory, "optimizer.pt"),
            )
        if self.scheduler is not None:
            torch.save(
                self.scheduler.state_dict(),
                os.path.join(save_directory, "scheduler.pt"),
            )
        if self.data_collator.tokenizer is not None:
            self.data_collator.tokenizer.to_json(
                os.path.join(save_directory, "vocab.json")
            )

    def wrap_model(self) -> Tuple[nn.Module, torch.optim.Optimizer]:
        if self.deepspeed:
            import deepspeed

            model_params = [p for p in self.model.parameters() if p.requires_grad]
            model, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=model_params,
                config="path_to_deepspeed_config.json",
            )
        else:
            model = self.model.to(self.device)
            if self.device_count > 1 and torch.cuda.is_available():
                model = nn.DataParallel(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
        return model, optimizer

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Unwrap a model.

        Useful when saving models that have been wrapped,
        e.g. using `nn.DataParallel` or `nn.DistributedDataParallel`.

        Parameters
        ----------
        model : nn.Module
            The model to unwrap.

        Returns
        -------
        nn.Module
            The unwrapped model.
        """
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        return model

    def place_inputs(self, collated: Dict):
        placed = {}
        for key, value in collated.items():
            value = value.to(self.device)
            placed[key] = value
        return placed

    @staticmethod
    def set_seed(seed: int):
        """
        Helper function for reproducible behavior to set the seed in
        ``random``, ``numpy``, and ``torch`` (if installed).

        Args:
            seed (`int`): The seed to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # safe even if cuda isn't available

    @staticmethod
    def print_train_log(
        steps: int,
        outputs: Union[MaskedLMOutput, dict],
        lr: float,
        num_train_steps: Optional[int] = None,
    ):
        if num_train_steps is not None:
            total_spaces = max(13, len(str(num_train_steps)))
            spaces = " " * (total_spaces - len(f"steps {steps} |"))
        else:
            spaces = ""
        log_str = f"step {steps}{spaces} | loss: {outputs['loss'].item():0.4f}"
        if "lm_loss" in outputs:
            log_str += f" | MLM loss: {outputs['lm_loss'].mean().item():0.4f}"
        if "router_z_loss" in outputs:
            log_str += (
                f" | router z-loss: {outputs['router_z_loss'].mean().item():0.4f}"
            )
        if "router_aux_loss" in outputs:
            log_str += (
                f" | router aux loss: {outputs['router_aux_loss'].mean().item():0.4f}"
            )
        log_str += f" | lr: {lr:0.6f}"
        print(log_str)

    @staticmethod
    def print_eval_log(
        eval_output: EvalOutput,
        num_train_steps: Optional[int] = None,
    ):
        if num_train_steps is not None:
            total_spaces = max(13, len(str(num_train_steps)))
            spaces = " " * (total_spaces - 12)
        else:
            spaces = " "
        print(f"<< EVAL >>{spaces}| loss: {eval_output.loss:.4f}", end="")
        if eval_output.metrics:
            for key, value in eval_output.metrics.items():
                print(f" | {key}: {value:.4f}", end="")
        print("")

#!/usr/bin/python
# filename: training_arguments.py

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
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import torch

from .training_utils import IntervalStrategy, OptimizerNames, SchedulerType


@dataclass
class TrainingArguments:
    """
    Arguments for training a model.

    Parameters
    ----------
    output_dir : str
        The output directory where the model predictions and checkpoints will be written.

    overwrite_output_dir : bool, default=False
        Overwrite the content of the output directory. Use this to continue training
        if output_dir points to a checkpoint directory.

    evaluation_strategy : Union[IntervalStrategy, str], default="no"
        The evaluation strategy to use. Possible values are:

            - `"no"`: No evaluation is done during training.
            - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
            - `"epoch"`: Evaluation is done at the end of each epoch.

    per_device_train_batch_size : int, default=8
        Batch size per GPU/MPS core or CPU for training.

    per_device_eval_batch_size : int, default=8
        Batch size per GPU/MPS core or CPU for evaluation.

    gradient_accumulation_steps : int, default=1
        Number of updates steps to accumulate before performing a backward/update pass.

        ..note::
            When using gradient accumulation, a step is only counted for logging, evaluation,
            and saving if it inclucdes a backward pass. Therefore, logging, evaluation, and
            save operations will be conducted every `gradient_accumulation_steps * xxx_step`
            training examples.

    learning_rate : float, default=5e-5
        The initial learning rate for AdamW.

    adam_beta1 : float, default=0.9
        Beta1 for AdamW optimizer.

    adam_beta2 : float, default=0.999
        Beta2 for AdamW optimizer.

    adam_epsilon : float, default=1e-8
        Epsilon for AdamW optimizer.

    max_grad_norm : float, default=1.0
        Max gradient norm  (for gradient clipping).

    max_epochs : int, default=3
        Total number of training epochs to perform.

    max_steps : int, default=-1
        If set to a positive number, the total number of training steps to perform.
        Overrides `num_train_epochs`. For a finite dataset, training is reiterated
        through the dataset (if all data is exhausted) until `max_steps` is reached.

    lr_scheduler_type : Union[SchedulerType, str], default="linear"
        The scheduler type to use.

    lr_scheduler_kwargs : Dict, default={}
        Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the
        cosine with hard restarts.

    warmup_ratio : float, default=0.0
        Linear warmup over warmup_ratio fraction of total steps.

    warmup_steps : int, default=0
        Linear warmup over warmup_steps.

    logging_dir : str, default=None
        Tensorboard log dir.

    logging_strategy : Union[IntervalStrategy, str], default="steps"
        The logging strategy to use.

    logging_first_step : bool, default=False
        Whether to log the first global_step.

    logging_steps : float, default=500
        Log every X updates steps. Should be an integer or a float in range `[0,1)`.
        If smaller than 1, will be interpreted as ratio of total training steps.

    save_strategy : Union[IntervalStrategy, str], default="steps"
        The saving strategy to use.

    save_steps : float, default=500
        Save every X updates steps. Should be an integer or a float in range `[0,1)`.
        If smaller than 1, will be interpreted as ratio of total training steps.

    save_total_limit : int, default=None
        Maximum number of checkpoints to save.

    use_cpu : bool, default=False
        Whether to use CPU instead of GPU.

    seed : int, default=42
        Random seed that will be set for reproducibility.

    data_seed : int, default=None
        Random seed to be used with data samplers. If not set, random generators
        for data sampling will use the same seed as `seed`. This can be used to
        ensure reproducibility of data sampling, independent of the model seed.

    fp16 : bool, default=False
        Whether to use 16-bit precision instead of 32-bit.

    ddp_backend : str, default="nccl"
        Backend to use for distributed training.

    dataloader_drop_last : bool, default=False
        Whether to drop the last incomplete batch.

    eval_steps : int, default=None
        Number of steps to run the evaluation for.

    run_name : str, default=None
        Name of the run.

    disable_tqdm : bool, default=False
        Whether to disable the tqdm progress bar.

    remove_unused_columns : bool, default=False
        Whether to remove unused columns from the dataset.

    label_names : List[str], default=None
        Names of the labels.

    accelerator_config : dict, default=None
        Configuration for the accelerator.

    deepspeed : str, default=None
        Configuration for DeepSpeed.

    label_smoothing_factor : float, default=0.0
        The factor by which to smooth the labels.

    optim : dict, default=None
        The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused,
        adamw_apex_fused, adamw_anyprecision, or adafactor.

    optim_args : dict, default=None
        Arguments for the optimizer.

    report_to : str, default=None
        The platform to report the results to: wandb, tensorboard, or none.

    dataloader_pin_memory : bool, default=True
        Whether to pin memory for the dataloaders.

    dataloader_num_workers : int, default=0
        The number of workers to use for the dataloaders.

    resume_from_checkpoint : str, default=None
        The path to the checkpoint from which to resume training.

    include_inputs_for_metrics : bool, default=False
        Whether or not the inputs will be passed to the `compute_metrics` function.
        This is intended for metrics that need inputs, predictions and references
        for scoring calculation in Metric class.

    """

    output_dir: str
    overwrite_output_dir: bool = False
    evaluation_strategy: Union[IntervalStrategy, str] = "no"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    do_eval: bool = False
    eval_delay: Optional[float] = 0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: Union[SchedulerType, str] = "linear"
    lr_scheduler_kwargs: Optional[Dict] = field(
        default_factory=dict,
    )
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    logging_dir: Optional[str] = None
    logging_strategy: Union[IntervalStrategy, str] = "steps"
    logging_first_step: bool = False
    logging_steps: float = 500
    save_strategy: Union[IntervalStrategy, str] = "steps"
    save_steps: float = 500
    save_total_limit: Optional[int] = None
    use_cpu: bool = False
    seed: int = 42
    data_seed: Optional[int] = None
    fp16: bool = False
    ddp_backend: Optional[str] = None
    dataloader_drop_last: bool = False
    eval_steps: Optional[float] = None
    run_name: Optional[str] = None
    disable_tqdm: Optional[bool] = None
    remove_unused_columns: Optional[bool] = True
    label_names: Optional[List[str]] = None
    # Do not touch this type annotation or it will stop working in CLI
    accelerator_config: Optional[str] = None
    # Do not touch this type annotation or it will stop working in CLI
    deepspeed: Optional[str] = None
    label_smoothing_factor: float = 0.0
    optim: Union[OptimizerNames, str] = "adamw_torch"
    optim_args: Optional[str] = None
    report_to: Optional[List[str]] = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    resume_from_checkpoint: Optional[str] = None
    include_inputs_for_metrics: bool = False

    def __post_init__(self):
        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        # LOGGING and EVAL
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True
        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (
            self.eval_steps is None or self.eval_steps == 0
        ):
            if self.logging_steps > 0:
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )
        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(
                f"logging strategy {self.logging_strategy} requires non-zero --logging_steps"
            )
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(
                    f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}"
                )
            self.logging_steps = int(self.logging_steps)
        if self.evaluation_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(
                    f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}"
                )
            self.eval_steps = int(self.eval_steps)

        # SAVING
        if self.save_strategy == IntervalStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(
                    f"--save_steps must be an integer if bigger than 1: {self.save_steps}"
                )
            self.save_steps = int(self.save_steps)

        # OPTIM
        self.optim = OptimizerNames(self.optim)

        # MISC
        if self.run_name is None:
            self.run_name = self.output_dir
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError(
                f"Invalid warmup_ratio: {self.warmup_ratio}. Must be between [0,1]"
            )
        if self.report_to is None:
            self.report_to = []
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]
        if self.use_cpu:
            self.dataloader_pin_memory = False

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {
            k: f"<{k.upper()}>" if k.endswith("_token") else v
            for k, v in self_as_dict.items()
        }

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        return self.per_device_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        return self.per_device_eval_batch_size * max(1, self.n_gpu)

    @cached_property
    def _setup_devices(self) -> "torch.device":
        """
        Setup devices (CPU, GPU, etc) for training.

        modified from: https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/training_args.py#L1931
        """
        self.distributed_state = None
        if torch.cuda.is_available() and not self.use_cpu:
            device = torch.device("cuda:0")
            self._n_gpu = torch.cuda.device_count()
            torch.cuda.set_device(device)
        elif torch.backends.mps.is_available() and not self.use_cpu:
            device = torch.device("mps")
            self._n_gpu = 1
        # elif self.deepspeed:
        #     # Need to do similar for Accelerator init
        #     os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
        #     self.distributed_state = PartialState(
        #         timeout=timedelta(seconds=self.ddp_timeout)
        #     )
        #     del os.environ["ACCELERATE_USE_DEEPSPEED"]
        #     self._n_gpu = 1
        else:
            device = torch.device("cpu")
            self._n_gpu = 0
        return device

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available
            but are not using distributed training. For distributed training, it will
            always be 1.
        """
        if not hasattr(self, "_n_gpu"):
            _ = self._setup_devices
        return self._n_gpu

    @property
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `torch.nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `torch.nn.DistributedDataParallel`).
        - `ParallelMode.TPU`: several TPU cores.
        """
        # requires_backends(self, ["torch"])
        # if is_torch_xla_available():
        #     return ParallelMode.TPU
        # elif is_sagemaker_mp_enabled():
        #     return ParallelMode.SAGEMAKER_MODEL_PARALLEL
        # elif is_sagemaker_dp_enabled():
        #     return ParallelMode.SAGEMAKER_DATA_PARALLEL
        # elif (
        #     self.distributed_state is not None
        #     and self.distributed_state.distributed_type != DistributedType.NO
        # ) or (self.distributed_state is None and self.local_rank != -1):
        #     return ParallelMode.DISTRIBUTED
        # elif self.n_gpu > 1:
        if self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps
            if self.warmup_steps > 0
            else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.init
        }

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
            # Handle the accelerator_config if passed
            # if is_accelerate_available() and isinstance(v, AcceleratorConfig):
            #     d[k] = v.to_dict()
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {
            **d,
            **{
                "train_batch_size": self.train_batch_size,
                "eval_batch_size": self.eval_batch_size,
            },
        }

        valid_types = [bool, int, float, str]
        valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
    SAGEMAKER_MODEL_PARALLEL = "sagemaker_model_parallel"
    SAGEMAKER_DATA_PARALLEL = "sagemaker_data_parallel"
    TPU = "tpu"

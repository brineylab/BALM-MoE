#!/usr/bin/python
# filename: training_utils.py

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


# import random
from enum import Enum
from functools import partial
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

# class ExplicitEnum(Enum):
#     def __init__(self, value):
#         self._value = value

#     def __repr__(self):
#         return f"{self.__class__.__name__}.{self.name}"

#     @classmethod
#     def _missing_(cls, value):
#         raise ValueError(
#             f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
#         )


# class IntervalStrategy(ExplicitEnum):
#     NO = "no"
#     STEPS = "steps"
#     EPOCHS = "epochs"


# class PaddingStrategy(ExplicitEnum):
#     LONGEST = "longest"
#     MAX_LENGTH = "max_length"
#     DO_NOT_PAD = "do_not_pad"


# class SchedulerType(ExplicitEnum):
#     LINEAR = "linear"
#     COSINE = "cosine"
#     COSINE_WITH_RESTARTS = "cosine_with_restarts"
#     POLYNOMIAL = "polynomial"
#     CONSTANT = "constant"
#     CONSTANT_WITH_WARMUP = "constant_with_warmup"
#     INVERSE_SQRT = "inverse_sqrt"
#     REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"


# class OptimizerNames(ExplicitEnum):
#     """
#     Stores the acceptable string identifiers for optimizers.
#     """

#     ADAMW_HF = "adamw_hf"
#     ADAMW_TORCH = "adamw_torch"
#     ADAMW_TORCH_FUSED = "adamw_torch_fused"
#     ADAMW_TORCH_XLA = "adamw_torch_xla"
#     ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
#     ADAMW_APEX_FUSED = "adamw_apex_fused"
#     ADAFACTOR = "adafactor"
#     ADAMW_ANYPRECISION = "adamw_anyprecision"
#     SGD = "sgd"
#     ADAGRAD = "adagrad"
#     ADAMW_BNB = "adamw_bnb_8bit"
#     ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
#     LION_8BIT = "lion_8bit"
#     LION = "lion_32bit"
#     PAGED_ADAMW = "paged_adamw_32bit"
#     PAGED_ADAMW_8BIT = "paged_adamw_8bit"
#     PAGED_LION = "paged_lion_32bit"
#     PAGED_LION_8BIT = "paged_lion_8bit"
#     RMSPROP = "rmsprop"
#     RMSPROP_BNB = "rmsprop_bnb"
#     RMSPROP_8BIT = "rmsprop_bnb_8bit"
#     RMSPROP_32BIT = "rmsprop_bnb_32bit"
#     GALORE_ADAMW = "galore_adamw"
#     GALORE_ADAMW_8BIT = "galore_adamw_8bit"
#     GALORE_ADAFACTOR = "galore_adafactor"
#     GALORE_ADAMW_LAYERWISE = "galore_adamw_layerwise"
#     GALORE_ADAMW_8BIT_LAYERWISE = "galore_adamw_8bit_layerwise"
#     GALORE_ADAFACTOR_LAYERWISE = "galore_adafactor_layerwise"


# def set_seed(seed: int):
#     """
#     Helper function for reproducible behavior to set the seed in
#     ``random``, ``numpy``, and ``torch`` (if installed).

#     Args:
#         seed (`int`): The seed to set.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # safe even if cuda isn't available


def _scheduler_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Taken from https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/optimization.py#L108

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = partial(
        _scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*):
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        labels: Union[np.ndarray, Tuple[np.ndarray]],
        logits: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        # inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.labels = labels
        self.logits = logits
        # self.inputs = inputs

    def __iter__(self):
        return iter((self.predictions, self.labels))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.labels
        elif idx == 2:
            return self.logits


class EvalOutput(NamedTuple):
    loss: float
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]

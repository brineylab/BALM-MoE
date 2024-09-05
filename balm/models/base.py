#!/usr/bin/python
# filename: base_model.py

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
import os
import re
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..config import BaseConfig


class BalmBase(nn.Module):
    """
    Base class for Balm models.
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, save_directory: str, max_shard_size: str = "10GB"):
        """
        Save the model's state dict to a directory.

        Parameters
        ----------
        save_directory : str
            The directory to save the model to.

        max_shard_size : str, optional, defaults to "10GB"
            The maximum size of each shard. If the model is smaller than the maximum size, it will be saved as a single file.
        """
        # make sure the save directory exists
        save_directory = os.path.abspath(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        # save the model config
        self.config.save_pretrained(save_directory)

        # unwrap the model
        model_to_save = unwrap_model(self)
        state_dict = model_to_save.state_dict()

        # shard and save the model
        shards, index = self._shard_checkpoint(
            state_dict, max_shard_size=max_shard_size
        )
        for shard_file, shard in shards.items():
            torch.save(shard, os.path.join(save_directory, shard_file))

        # save the index
        if index is not None:
            save_index_file = os.path.join(save_directory, "model.pt.index.json")
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *model_args,
        config: Optional[Union[str, BaseConfig, dict]] = None,
    ):
        """
        Load a pretrained model.

        Parameters
        ----------
        model_path : str
            The path to a diretory containing the pretrained model.
            Model file should be named ``"model.pt"``.

        model_args : tuple, optional
            Additional arguments that will be passed directly to the model's __init__ method.

        config : Optional[Union[str, BaseConfig, dict]], optional
            Alternate configuration object or path to an alternate configuration file.
            If not provided, the configuration (``config.json``) will be loaded from the
            model directory.

        Returns
        -------
        cls
            The loaded model.
        """
        # config
        if config is None:
            config = os.path.join(model_path, "config.json")
        config = cls.config_class.from_pretrained(config)

        # model
        model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file (model.pt) not found in the supplied model directory: {model_path}"
            )
        model = cls(config=config, *model_args)
        state_dict_to_load = torch.load(model_path)
        # if state_dict keys start with "module.", the model wasn't unwrapped before saving
        # and the keys won't match the madel we're trying to load into
        while all(k.startswith("module.") for k in state_dict_to_load.keys()):
            state_dict_to_load = {k[7:]: v for k, v in state_dict_to_load.items()}
        extra_keys = model.load_state_dict(state_dict_to_load, strict=False)

        # check to see whether there are any extra (unexpected or missing) keys
        # unexpected keys are keys in the saved state dict that are not in the model
        # missing keys are keys in the model that are not in the saved state dict
        if len(extra_keys.unexpected_keys) > 0:
            unexpected_keys = "\n  - " + "\n  - ".join(extra_keys.unexpected_keys)
            print(
                f"Some weights of the model checkpoint at {model_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\nThis IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BalmForSequenceClassification model from a"
                " BalmForMaskedLM model).\nThis IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (e.g. initializing a BalmForSequenceClassification model from a BalmForSequenceClassification model).\n"
            )
        else:
            print(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )
        if len(extra_keys.missing_keys) > 0:
            missing_keys = "\n  - " + "\n  - ".join(extra_keys.missing_keys)
            print(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {model_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a downstream task to be able to use it for predictions and inference."
            )
        else:
            print(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {model_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        return model

    def _shard_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        max_shard_size: Union[int, str] = "10GB",
        weights_name: str = "model.pt",
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
        given size.

        from 🤗 -- https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/modeling_utils.py#L333

        The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
        optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
        limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
        [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

        .. note::
            If one of the model's weight is bigger than `max_shard_size`, it will end up in its own sub-checkpoint which will
            have a size greater than `max_shard_size`.

        Parameters
        ----------
        state_dict : Dict[str, torch.Tensor]
            The state dictionary of a model to save.

        max_shard_size : int or str, optional, defaults to "10GB"
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).

        weights_name : str, optional, defaults to "model.pt"
            The name of the model save file.

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]
            The sharded model state dictionary and the index. If the model is only one shard, the index is `None`.

        """
        max_shard_size = self.convert_file_size_to_int(max_shard_size)
        sharded_state_dicts = [{}]
        last_block_size = 0
        total_size = 0

        for key, weight in state_dict.items():
            weight_size = weight.numel() * self.dtype_byte_size(weight.dtype)

            # if this weight is going to take us over the maximum shard size, we split
            # (but only if the current shard has at least one weight)
            # this accounts for cases where a single weight exceeds the maximum shard size
            if (
                last_block_size + weight_size > max_shard_size
                and len(sharded_state_dicts[-1]) > 0
            ):
                sharded_state_dicts.append({})
                last_block_size = 0

            sharded_state_dicts[-1][key] = weight
            last_block_size += weight_size
            total_size += weight_size

        # if we only have one shard, we return it with index as None
        if len(sharded_state_dicts) == 1:
            return {weights_name: sharded_state_dicts[0]}, None

        # otherwise, build the shard index
        weight_map = {}
        shards = {}
        extension = weights_name.split(".")[-1]
        for idx, shard in enumerate(sharded_state_dicts):
            shard_file = weights_name.replace(
                f".{extension}",
                f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.{extension}",
            )
            shards[shard_file] = shard
            for key in shard.keys():
                weight_map[key] = shard_file

        # add metadata
        metadata = {"total_size": total_size}
        index = {"metadata": metadata, "weight_map": weight_map}
        return shards, index

    @staticmethod
    def dtype_byte_size(dtype) -> int:
        """
        Returns the size (in bytes) occupied by one parameter of type `dtype`.

        from 🤗 -- https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/modeling_utils.py#L313

        Example:

        ```py
        >>> dtype_byte_size(torch.float32)
        4
        ```

        Parameters
        ----------
        dtype : torch.dtype
            The dtype to get the size of.

        Returns
        -------
        int
            The size in bytes.
        """
        if dtype == torch.bool:
            return 1 / 8
        bit_search = re.search(r"[^\d](\d+)$", str(dtype))
        if bit_search is None:
            raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
        bit_size = int(bit_search.groups()[0])
        return bit_size // 8

    @staticmethod
    def convert_file_size_to_int(size: Union[int, str]) -> int:
        """
        Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

        from 🤗 -- https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/utils/hub.py#L941

        Example:
        ```py
        >>> convert_file_size_to_int("1MiB")
        1048576
        ```

        Parameters
        ----------
        size : int or str
            The size to convert. Will be directly returned if an `int`.

        Returns
        -------
        int
            The size in bytes.
        """
        if isinstance(size, int):
            return size
        if size.upper().endswith("GIB"):
            return int(size[:-3]) * (2**30)
        if size.upper().endswith("MIB"):
            return int(size[:-3]) * (2**20)
        if size.upper().endswith("KIB"):
            return int(size[:-3]) * (2**10)
        if size.upper().endswith("GB"):
            int_size = int(size[:-2]) * (10**9)
            return int_size // 8 if size.endswith("b") else int_size
        if size.upper().endswith("MB"):
            int_size = int(size[:-2]) * (10**6)
            return int_size // 8 if size.endswith("b") else int_size
        if size.upper().endswith("KB"):
            int_size = int(size[:-2]) * (10**3)
            return int_size // 8 if size.endswith("b") else int_size
        raise ValueError(
            "`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'."
        )


def unwrap_model(model: nn.Module) -> nn.Module:
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
        return unwrap_model(model.module)
    return model

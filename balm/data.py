#!/usr/bin/python
# filename: data.py

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


import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch

from .tokenizer import BatchEncoding, TokenizerBase

__all__ = [
    "Dataset",
    "DatasetDict",
    "DataCollator",
    "load_dataset",
]


class Dataset:
    def __init__(self, data: Dict[str, Iterable[str]]):
        self.data = pl.DataFrame(data)

    def __getitem__(self, index: Union[int, str]):
        if isinstance(index, str) and index in self.data.columns:
            return self.data[index].to_list()
        elif isinstance(index, int):
            d = self.data[index].to_dict(as_series=False)
            return {k: v[0] for k, v in d.items()}
        else:
            raise ValueError(f"Index {index} is not valid")

    def __setitem__(self, index: str, value: Any):
        if len(value) != self.num_rows:
            raise ValueError(
                f"Value length {len(value)} does not match dataset length {self.num_rows}"
            )
        self.data = self.data.with_columns(pl.Series(name=index, values=value))

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset.
        """
        return self.data.shape

    @property
    def num_rows(self) -> int:
        """
        Returns the number of rows in the dataset.
        """
        return self.data.shape[0]

    def remove_columns(self, columns: Union[str, Iterable[str]]):
        """
        Removes the specified columns from the dataset.
        """
        if isinstance(columns, str):
            columns = [columns]
        self.data = self.data.drop(columns)

    def rename_columns(self, columns: Dict[str, str]):
        """
        Renames the specified columns in the dataset.
        """
        self.data = self.data.rename(columns)

    def clone(self):
        """
        Returns a clone of the dataset.
        """
        return self.__class__(self.data.clone())


class DatasetDict:
    def __init__(self, data: Dict[str, Dataset]):
        self.dataset_dict = data

    def __getitem__(self, index: str):
        return self.dataset_dict[index]

    def __repr__(self):
        repr = "DatasetDict\n-----------\n"
        for name, dataset in self.dataset_dict.items():
            repr += f"  {name}\n"
            repr += f"    num_rows: {dataset.num_rows}\n"
            repr += f"    columns: {dataset.data.columns}\n"
        return repr

    def __str__(self):
        return self.__repr__()

    def keys(self):
        return self.dataset_dict.keys()

    def values(self):
        return self.dataset_dict.values()

    def items(self):
        return self.dataset_dict.items()

    def map(
        self,
        func: Callable,
        remove_columns: Optional[Union[str, Iterable[str]]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
    ):
        cloned_data = {k: v.clone() for k, v in self.items()}
        dataset_dict = self.__class__(cloned_data)
        for dataset in dataset_dict.values():
            result = func(dataset)
            if not isinstance(result, (dict, BatchEncoding)):
                raise ValueError(
                    f"Mapping function returned an object of type {type(result)}, but must return either a dictionary or BatchEncoding object"
                )
            for key, value in result.items():
                dataset[key] = value
            if remove_columns is not None:
                if isinstance(remove_columns, str):
                    remove_columns = [remove_columns]
                dataset.remove_columns(remove_columns)
            if rename_columns is not None:
                dataset.rename_columns(rename_columns)
        return dataset_dict


class DataCollator:
    def __init__(
        self,
        tokenizer: TokenizerBase,
        mlm: bool = True,
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(
        self,
        examples: List[Union[List[int], Any, Dict[str, Any]]],
        mask_probs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Collate a batch of examples.

        Parameters
        ----------
        examples : List[Union[List[int], Any, Dict[str, Any]]]
            The batch of examples to collate.

        mask_probs : Optional[str]
            The name of a key in the examples dictionary that contains the masking probabilities
            for MLM training. If not specified, examples will be uniformly masked at ``self.mlm_probability``.

        Returns
        -------
        Dict[str, Any]
            The collated batch.

        """
        # convert to tensors if necessary
        if isinstance(examples, dict):
            batch = examples
        elif isinstance(examples, torch.Tensor):
            batch = {"input_ids": examples}
        else:
            if isinstance(examples[0], (list, tuple, np.ndarray)):
                examples = [torch.tensor(e, dtype=torch.long) for e in examples]
            # batch = {"input_ids": torch.stack(examples)}
            batch = {"input_ids": torch.stack([d["input_ids"] for d in examples])}

        # MLM masking
        if self.mlm:
            # TODO: implement mask_probs
            # not sure how they'll be formatted -- as a stacked tensor, a list of tensors, or a list of CIGAR-like strings
            batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"])
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        # key padding mask
        kp_mask = torch.zeros_like(batch["input_ids"])  # 1 for non-pad tokens
        kp_mask[batch["input_ids"] == self.tokenizer.pad_idx] = 1  # 0 for pad tokens
        batch["key_padding_mask"] = kp_mask.bool()
        return batch

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        mask_probs: Optional[str] = None,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val) for val in labels.tolist()
            ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )

        random_tokens = np.random.choice(
            self.tokenizer.all_nonspecial_tokens_idx,
            size=inputs.numel(),
            replace=True,
        )
        random_tokens = torch.tensor(random_tokens, dtype=torch.long).view(inputs.shape)
        # random_tokens = torch.multinomial(
        #     self.tokenizer.all_nonspecial_tokens_idx,
        #     labels.numel,
        #     replacement=True,
        #     dtype=torch.long,
        # ).view(labels.shape)
        inputs[indices_random] = random_tokens[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def load_dataset(
    path: str,
    data_files: Dict[str, str],
    strip_lines: bool = True,
    preprocess_fn: Optional[Callable] = None,
):
    dataset_dict = {}
    for name, files in data_files.items():
        if os.path.isdir(files):
            files = [os.path.join(files, f) for f in os.listdir(files)]
        elif os.path.isfile(files):
            files = [files]
        else:
            raise ValueError(f"Invalid file or directory: {files}")
        data = []
        for file in files:
            with open(file, "r") as f:
                file_data = f.readlines()
                if strip_lines:
                    file_data = [line.strip() for line in file_data]
                if preprocess_fn is not None:
                    file_data = [preprocess_fn(line) for line in file_data]
                data.extend(file_data)
        dataset_dict[name] = Dataset({path: data})
    return DatasetDict(dataset_dict)

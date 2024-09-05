#!/usr/bin/python
# filename: embedding.py

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


import torch
from torch import nn

__all__ = [
    "RelativePositionalEmbedding",
    "RotaryPositionalEmbedding",
]


class RelativePositionalEmbedding(nn.Module):
    """
    Relative positional embeddings, as initially described in the
    `Relative Position Embeddings for Transformers`_ paper.

    Parameters
    ----------
    embed_dim: int
        The embedding dimension.

    max_length: int
        The maximum length of the input tensor.

    .. _Relative Position Embeddings for Transformers:
        https://arxiv.org/abs/1803.02155

    """

    def __init__(self, embed_dim: int, max_length: int):
        super(RelativePositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply relative positional embeddings to the input tensor.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            The input tensor with relative positional embeddings applied. The shape is (batch_size, seq_len, dim).
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embeddings, as initially described in the
    `RoFormer: Enhanced Transformer with Rotary Position Embeddings`_ paper.

    Parameters
    ----------
    embed_dim: int
        The embedding dimension.

    max_length: int
        The maximum length of the input tensor.

    .. _RoFormer: Enhanced Transformer with Rotary Position Embeddings:
        https://arxiv.org/abs/2104.09864
    """

    def __init__(self, embed_dim: int, max_length: int):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim)
        )

    def get_positional_embeddings(self, x: torch.Tensor):
        """
        Generates the sinusoidal positional embeddings.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            The positional embeddings. The shape is (seq_len, dim).
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

    def apply_rotary_embeddings(self, x: torch.Tensor):
        """
        Applies rotary embeddings to the input tensor x.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            The input tensor with rotary embeddings applied. The shape is (batch_size, seq_len, dim).

        """
        pos_emb = self.get_positional_embeddings(x).to(x.device)
        s, c = pos_emb[:, : self.embed_dim // 2], pos_emb[:, self.embed_dim // 2 :]
        x1, x2 = x[..., : self.embed_dim // 2], x[..., self.embed_dim // 2 :]
        x_rot = torch.cat(((x1 * c) + (x2 * s), (-x1 * s) + (x2 * c)), dim=-1)
        return x_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to the input tensor.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            The input tensor with rotary positional embeddings applied. The shape is (batch_size, seq_len, dim).
        """
        return self.apply_rotary_embeddings(x)

#!/usr/bin/python
# filename: loss.py

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
from torch.nn import functional as F

__all__ = ["router_z_loss", "router_load_balancing_loss"]


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the router z-loss.

    The router z-loss was introduced in `Designing Effective Sparse Expert Models`_.
    It encourages router logits to remain small in an effort to improve stability.


    Parameters
    ----------
    router_logits : float
        Input logits of shape [batch_size, sequence_length, num_experts]

    Returns
    -------
    torch.Tensor
        The z-loss for the router


    .. _Designing Effective Sparse Expert Models:
        https://arxiv.org/abs/2202.08906
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def router_load_balancing_loss(
    router_probs: torch.Tensor, expert_indices: torch.Tensor
) -> torch.Tensor:
    """
    Computes the auxiliary load balancing loss.

    See the `Switch Transformer manuscript`_ for more details. This function
    implements the loss function presented in equations (4) - (6) of the paper.
    It aims at penalizing cases where the routing between experts is too unbalanced.


    Parameters:
    -----------
    router_probs : torch.Tensor
        Probability assigned to each expert per token.
        Shape: [batch_size, seqeunce_length, num_experts].

    expert_indices : torch.Tensor
        Indices tensor of identifying the selected expert for a given token.
        Shape: [batch_size, seqeunce_length]

    Returns
    -------
    torch.Tensor
        The auxiliary load balancing loss for the router


    .. _Switch Transformer manuscript:
        https://arxiv.org/abs/2101.03961
    """
    num_experts = router_probs.shape[-1]
    if expert_indices.dtype != torch.int64:  # F.one_hot fails if not int64
        expert_indices = expert_indices.to(torch.int64)
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)
    # expert mask
    expert_mask = F.one_hot(expert_indices, num_experts)
    expert_mask = torch.max(expert_mask, axis=-2).values
    expert_mask = expert_mask.to(torch.float32)  # torch.mean needs float32
    # compute aux loss
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert
    ) * (num_experts**2)


# import torch
# from torch import nn

# __all__ = ["router_z_loss", "router_load_balancing_loss"]


# def router_z_loss(router_logits: torch.Tensor) -> float:
#     """
#     Computes the router z-loss.

#     The router z-loss was introduced in `Designing Effective Sparse Expert Models`_.
#     It encourages router logits to remain small in an effort to improve stability.


#     Parameters:
#     -----------
#     router_logits : float
#         Input logits of shape [batch_size, sequence_length, num_experts]

#     Returns:
#     --------
#         Scalar router z-loss


#     .. _Designing Effective Sparse Expert Models:
#         https://arxiv.org/abs/2202.08906
#     """
#     num_groups, tokens_per_group, _ = router_logits.shape
#     log_z = torch.logsumexp(router_logits, dim=-1)
#     z_loss = log_z**2
#     return torch.sum(z_loss) / (num_groups * tokens_per_group)


# def router_load_balancing_loss(
#     router_probs: torch.Tensor, expert_indices: torch.Tensor
# ) -> float:
#     """
#     Computes the auxiliary load balancing loss.

#     See the `Switch Transformer manuscript`_ for more details. This function
#     implements the loss function presented in equations (4) - (6) of the paper.
#     It aims at penalizing cases where the routing between experts is too unbalanced.


#     Parameters:
#     -----------
#     router_probs : torch.Tensor
#         Probability assigned to each expert per token.
#         Shape: [batch_size, seqeunce_length, num_experts].

#     expert_indices : torch.Tensor
#         Indices tensor of identifying the selected expert for a given token.
#         Shape: [batch_size, seqeunce_length]

#     Returns:
#         The auxiliary loss.


#     .. _Switch Transformer manuscript:
#         https://arxiv.org/abs/2101.03961
#     """
#     num_experts = router_probs.shape[-1]

#     # cast the expert indices to int64, otherwise one-hot encoding will fail
#     if expert_indices.dtype != torch.int64:
#         expert_indices = expert_indices.to(torch.int64)

#     if len(expert_indices.shape) == 2:
#         expert_indices = expert_indices.unsqueeze(2)

#     expert_mask = nn.functional.one_hot(expert_indices, num_experts)

#     # For a given token, determine if it was routed to a given expert.
#     expert_mask = torch.max(expert_mask, axis=-2).values

#     # cast to float32 otherwise mean will fail
#     expert_mask = expert_mask.to(torch.float32)
#     tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

#     router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
#     return torch.mean(
#         tokens_per_group_and_expert * router_prob_per_group_and_expert
#     ) * (num_experts**2)

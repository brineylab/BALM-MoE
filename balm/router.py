#!/usr/bin/python
# filename: router.py

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


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    # "RouterBase",
    "TopKRouter",
    "ExpertChoiceRouter",
]


class RouterBase(nn.Module):
    """
    Base class for routers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity: int,
        dtype: str = "float32",
        bias: bool = False,
        jitter: float = 0.0,
        num_routable_experts: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dtype = getattr(torch, dtype)
        self.bias = bias
        self.jitter = jitter
        self.classifier = nn.Linear(
            self.embed_dim,
            num_routable_experts
            if num_routable_experts is not None
            else self.num_experts,
            bias=self.bias,
            dtype=self.dtype,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _compute_router_probabilities(
        self,
        x: torch.Tensor,
        dim: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes router probabilities from input hidden states.

        Parameters:
        -----------
        x : torch.Tensor
            Tensor of shape (batch_size, sequence_length, hidden_dim) from which
            router probabilities are computed.

        dim : int, optional
            Dimension along which to compute the softmax. The default is -1, which corresponds
            to token-choice routing. For expert choice routing, this should be -2.

        Returns:
        --------
        router_probabilities : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) corresponding to
            the probabilities for each token and expert. Used for routing tokens to experts.

        router_logits : torch.Tensor
            Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding
            to raw router logits. This is used for computing router z-loss.
        """
        # float32 is used to ensure stability. See https://arxiv.org/abs/2101.03961.
        self.input_dtype = x.dtype
        x = x.to(self.dtype)
        if self.jitter > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter, 1.0 + self.jitter)
        logits = self.classifier(x)  # (batch, seq_len, num_experts)
        probabilities = F.softmax(logits, dim=dim, dtype=self.dtype).to(
            self.input_dtype
        )
        return probabilities, logits


class TopKRouter(RouterBase):
    """
    This router uses the "token choice of top-k experts" strategy. For example, if k=1, this
    replicates the top-1 routing strategy introduced in the `Switch Transformers`_ paper.
    Alternatively, if k=2, this replicates the top-2 routing strategy introduced in the `GShard`_
    paper. Tokens are routed to their expert of choice until the expert's `expert_capacity` is
    reached. Shared experts, which process all tokens, are implemented as described in the
    `DeepSeqMoE`_ paper.

    .. note::
        There is no guarantee that each token will be processed by an expert,
        or that every expert will receive at least one token.

    If tokens are routed to an expert which is above capacity, they are not processed by any expert
    and their hidden states are passed to the subsequent layer unchanged.


    Parameters:
    -----------
    embed_dim : int
        Embedding dimension.

    num_experts : int
        Number of experts.

    expert_capacity : int
        Maximum number of tokens that can be routed to each expert.

    top_k : int, optional
        Number of top experts to route each token to. The default is 1.

    num_shared_experts : int, optional
        Number of shared experts that process all tokens. The default is 0.

    dtype : str, optional
        Data type to use for router probabilities. The default is "float32".

    bias : bool, optional
        Whether to add bias to the router classifier. The default is ``False``.

    jitter : float, optional
        Amount of jitter to add to the router probabilities. The default is ``0.0``.

    ignore_padding_tokens : bool, optional
        Whether to ignore padding tokens when computing router probabilities.
        The default is ``True``.


    .. _Switch Transformers:
        https://arxiv.org/abs/2101.03961

    .. _GShard:
        https://arxiv.org/abs/2006.16668

    .. _DeepSeqMoE:
        https://arxiv.org/abs/2401.06066
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity: int,
        top_k: int = 1,
        num_shared_experts: int = 0,
        dtype: str = "float32",
        bias: bool = False,
        jitter: float = 0.0,
        ignore_padding_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            dtype=dtype,
            bias=bias,
            jitter=jitter,
            num_routable_experts=num_experts - num_shared_experts,
        )
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.ignore_padding_tokens = ignore_padding_tokens

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Token choice of top-k experts, with optional shared experts processing all tokens.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        expert_mask : torch.Tensor
            Binary mask tensor of shape (batch_size, sequence_length, num_experts)
            indicating which experts the token should be routed to (including shared experts).

        router_probabilities : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router probabilities.

        router_logits : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router logits.
        """
        num_routable_experts = self.num_experts - self.num_shared_experts

        # router
        router_probs, router_logits = self._compute_router_probabilities(x)
        _, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        expert_mask = F.one_hot(topk_indices, num_classes=num_routable_experts).sum(
            dim=-2
        )

        # mask tokens if their desired experts are above capacity
        token_priority = torch.cumsum(expert_mask, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_mask = expert_mask * expert_capacity_mask

        # shared experts
        if self.num_shared_experts > 0:
            # include shared experts in the expert mask (first N experts are shared experts)
            shared_expert_mask = torch.ones_like(
                router_probs[..., : self.num_shared_experts]
            )
            expert_mask = torch.cat((shared_expert_mask, expert_mask), dim=-1)
            # add shared experts to router probs
            shared_expert_probs = torch.ones_like(
                router_probs[..., : self.num_shared_experts]
            )
            router_probs = torch.cat((shared_expert_probs, router_probs), dim=-1)

        return expert_mask, router_probs, router_logits


class ExpertChoiceRouter(RouterBase):
    """
    This router uses the "expert choice of top-k tokens" strategy, as originally described
    in the `Mixture-of-Experts with Expert Choice Routing`_ paper. This automatically
    balances the number of tokens processed by each expert, and eliminates the
    need for an auxiliary (load-balancing) router loss.

    .. note::
        There is no guarantee that each token will be processed by an expert. In fact,
        one of the primary benefits of expert choice routing is thought to be their
        ability to heterogeneously devote computation to a subset of highly complex/difficult
        tokens.

    If tokens are not selected by an expert, their hidden states are passed to the
    subsequent layer unchanged.

    Parameters:
    -----------
    embed_dim : int
        Embedding dimension.

    num_experts : int
        Number of experts.

    expert_capacity : int
        Maximum number of tokens that can be routed to each expert.

    num_shared_experts : int, optional
        Number of shared experts that process all tokens. The default is 0.

    dtype : str, optional
        Data type to use for router probabilities. The default is "float32".

    bias : bool, optional
        Whether to add bias to the router classifier. The default is ``False``.

    jitter : float, optional
        Amount of jitter to add to the router probabilities. The default is ``0.0``.

    ignore_padding_tokens : bool, optional
        Whether to ignore padding tokens when computing router probabilities.
        The default is ``True``.

    .. _Mixture-of-Experts with Expert Choice Routing:
        https://arxiv.org/abs/2202.09368
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity: int,
        num_shared_experts: int = 0,
        dtype: str = "float32",
        bias: bool = False,
        jitter: float = 0.0,
        ignore_padding_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            dtype=dtype,
            bias=bias,
            jitter=jitter,
            num_routable_experts=num_experts - num_shared_experts,
        )
        self.num_shared_experts = num_shared_experts
        self.ignore_padding_tokens = ignore_padding_tokens

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert choice of top-k tokens, with optional shared experts that process all tokens.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        expert_mask : torch.Tensor
            Binary mask tensor of shape (batch_size, sequence_length, num_experts) indicating
            which tokens are selected for each expert and which are processed by shared experts.

        router_probabilities : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router probabilities.

        router_logits : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router logits.
        """
        router_probs, router_logits = self._compute_router_probabilities(x, dim=-2)
        expert_mask = torch.zeros_like(router_probs)

        # Select top-k tokens for each expert
        for i in range(self.num_experts - self.num_shared_experts):
            _, top_k_indices = torch.topk(
                router_probs[..., i], k=self.expert_capacity, dim=1
            )
            expert_mask[:, :, i].scatter_(1, top_k_indices, 1)

        # shared experts
        if self.num_shared_experts > 0:
            # include shared experts in the expert mask (first N experts are shared experts)
            shared_expert_mask = torch.ones_like(
                router_probs[..., : self.num_shared_experts]
            )
            expert_mask = torch.cat((shared_expert_mask, expert_mask), dim=-1)
            # add shared experts to router probs
            shared_expert_probs = torch.ones_like(
                router_probs[..., : self.num_shared_experts]
            )
            router_probs = torch.cat((shared_expert_probs, router_probs), dim=-1)

        return expert_mask, router_probs, router_logits

#!/usr/bin/python
# filename: balm_moe_config.py

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


from typing import Optional

from .base import BaseConfig


class BalmMoEConfig(BaseConfig):
    def __init__(
        self,
        embed_dim: int = 320,
        ffn_dim: int = 1280,
        num_layers: int = 6,
        num_heads: int = 20,
        num_experts: int = 8,
        max_length: int = 320,
        vocab_size: int = 33,
        expert_capacity: Optional[int] = None,
        expert_capacity_multiplier: float = 1.5,
        num_shared_experts: int = 0,
        activation: str = "swiglu",
        positional_embedding_type: str = "rotary",
        pre_norm: bool = True,
        router_z_loss_coef: float = 1e-3,
        router_aux_loss_coef: float = 1e-2,
        alternate_sparsity: bool = False,
        router_top_k: int = 1,
        router_bias: bool = False,
        router_jitter: float = 0.0,
        router_dtype: str = "float32",
        router_ignore_padding_tokens: bool = True,
        expert_choice_router: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        expert_ffn_dropout: float = 0.0,
        token_embedding_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        padding_idx: int = 0,
        # classification head
        num_labels: int = 2,
        classifier_dropout: float = 0.0,
        classifier_activation: str = "tanh",
    ):
        """
        Configuration for the BalmMoE model. Default parameters are similar to the 8M parameter ESM-2 model.

        Parameters
        ----------
        embed_dim : int, default=320
            The dimension of the token embeddings.

        ffn_dim : int, default=1280
            The dimension of the feed-forward network.

        num_layers : int, default=6
            The number of layers in the transformer.

        num_heads : int, default=20
            The number of heads in the transformer.

        num_experts : int, default=8
            The number of experts in the transformer.

        max_length : int, default=320
            The maximum length of the input sequence.

        vocab_size : int, default=33
            The vocabulary size.

        expert_capacity : int, optional
            The capacity of each expert. If not provided, it will be calculated as `max_length / num_experts * expert_capacity_multiplier`.

        expert_capacity_multiplier : float, default=1.25
            The multiplier for the expert capacity.

        num_shared_experts : int, default=0
            The number of shared experts in the transformer.

        activation : str, default="gelu"
            The activation function to use for the experts. Options are "relu" and "gelu".

        router_z_loss_coef : float, default=0.001
            The coefficient for the router z loss.

        router_aux_loss_coef : float, default=0.001
            The coefficient for the router auxiliary loss.

        alternate_sparsity : bool, default=False
            Whether to use alternate sparsity for the router.

        router_top_k : int, default=1
            The top k to use for the router.

        router_bias : bool, default=False
            Whether to use a bias for the router.

        router_jitter : float, default=0.0
            The jitter to use for the router.

        router_dtype : str, default="float32"
            The dtype to use for the router. Options are "float32" and "float16".

        router_ignore_padding_tokens : bool, default=True
            Whether to ignore padding tokens for the router.

        dropout : float, default=0.1
            The dropout to use for the transformer.

        attention_dropout : float, default=0.0
            The dropout to use for the attention.

        ffn_dropout : float, default=0.0
            The dropout to use for the experts.

        token_embedding_dropout : float, default=0.0
            The dropout to use for the token embeddings.

        layer_norm_eps : float, default=1e-5
            The epsilon to use for the layer normalization.

        padding_idx : int, default=0
            The index to use for the padding tokens.

        num_labels : int, default=2
            The number of labels for the classification head.

        classifier_dropout : float, default=0.0
            The dropout to use for the classification head.

        classifier_activation : str, default="tanh"
            The activation function to use for the classification head. Options are "tanh" and "softmax".
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_experts = int(num_experts)
        self.max_length = int(max_length)
        self.vocab_size = int(vocab_size)
        self.expert_capacity_multiplier = float(expert_capacity_multiplier)
        self.expert_capacity = (
            int(expert_capacity)
            if expert_capacity is not None
            else int(max_length / num_experts * self.expert_capacity_multiplier)
        )
        self.num_shared_experts = int(num_shared_experts)
        if positional_embedding_type.lower() not in ["rotary", "relative"]:
            raise ValueError(
                f"Invalid positional embedding type: {positional_embedding_type}. Options are 'rotary' or 'relative'."
            )
        self.positional_embedding_type = positional_embedding_type.lower()
        if activation.lower() not in ["swiglu", "relu", "gelu"]:
            raise ValueError(
                f"Invalid FFN activation: {activation}. Options are 'swiglu', 'relu', or 'gelu'."
            )
        self.activation = activation.lower()
        self.pre_norm = bool(pre_norm)
        self.router_z_loss_coef = float(router_z_loss_coef)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.alternate_sparsity = alternate_sparsity
        self.router_top_k = int(router_top_k)
        self.router_bias = bool(router_bias)
        self.router_jitter = float(router_jitter)
        self.router_dtype = router_dtype.lower()
        self.router_ignore_padding_tokens = bool(router_ignore_padding_tokens)
        self.expert_choice_router = bool(expert_choice_router)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.expert_ffn_dropout = float(expert_ffn_dropout)
        self.token_embedding_dropout = float(token_embedding_dropout)
        self.layer_norm_eps = float(layer_norm_eps)
        self.padding_idx = int(padding_idx)
        # classification head
        self.num_labels = int(num_labels)
        self.classifier_dropout = float(classifier_dropout)
        if classifier_activation.lower() not in ["tanh", "relu", "gelu"]:
            raise ValueError(
                f"Invalid classification head activation: {classifier_activation}. Options are 'tanh', 'relu', or 'gelu'."
            )
        self.classifier_activation = classifier_activation.lower()

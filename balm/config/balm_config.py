#!/usr/bin/python
# filename: balm_config.py

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


from .base import BaseConfig


class BalmConfig(BaseConfig):
    def __init__(
        self,
        embed_dim: int = 320,
        ffn_dim: int = 1280,
        num_layers: int = 6,
        num_heads: int = 20,
        num_experts: int = 8,
        max_length: int = 320,
        vocab_size: int = 33,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        token_embedding_dropout: float = 0.0,
        positional_embedding_type: str = "rotary",
        pre_norm: bool = True,
        activation: str = "swiglu",
        layer_norm_eps: float = 1e-5,
        padding_idx: int = 0,
        # classification head
        num_labels: int = 2,
        classifier_dropout: float = 0.0,
        classifier_activation: str = "tanh",
    ):
        """
        Configuration for the Balm model. Default parameters are similar to the 8M parameter ESM-2 model.

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

        dropout : float, default=0.1
            The dropout rate. Applied immediately before adding the residual connection.

        attention_dropout : float, default=0.0
            The dropout rate for the attention layer.

        token_embedding_dropout : float, default=0.0
            The dropout rate for the token embedding layer.

        positional_embedding_type : str, default="rotary"
            The type of positional embedding to use. Options are "rotary" or "relative".

        pre_norm : bool, default=True
            Whether to use pre-norm or post-norm.

        activation : str, default="swiglu"
            The activation function to use in the feed-forward network. Options are "swiglu", "relu", or "gelu".

        layer_norm_eps : float, default=1e-5
            The epsilon value for the layer normalization.

        padding_idx : int, default=0
            The index of the padding token.

        num_labels : int, default=2
            The number of labels for the sequence classification head. Only used for BalmForSequenceClassification models.

        classifier_dropout : float, default=0.0
            The dropout rate for the sequence classification head. Only used for BalmForSequenceClassification models.

        classifier_activation : str, default="tanh"
            The activation function to use in the sequence classification head. Only used for BalmForSequenceClassification models.
            Options are "tanh", "relu", or "gelu".
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_experts = int(num_experts)
        self.max_length = int(max_length)
        self.vocab_size = int(vocab_size)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.token_embedding_dropout = float(token_embedding_dropout)
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

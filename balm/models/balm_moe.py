#!/usr/bin/python
# filename: balm_moe.py

#
# Copyright (c) 2023 Bryan Briney
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

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import BalmMoEConfig
from ..loss import router_load_balancing_loss, router_z_loss
from ..modules import (
    BalmClassificationHead,
    BalmLMHead,
    ClassifierOutput,
    DenseTransformerLayer,
    MaskedLMOutput,
    SparseTransformerLayer,
)
from .base import BalmBase

__all__ = [
    "BalmMoEModel",
    "BalmMoEForMaskedLM",
    "BalmMoEForSequenceClassification",
]


class BalmMoEModel(BalmBase):
    """
    BALM Mixture-of-Experts model

    Parameters
    ----------
    config: BalmMoEConfig
        Configuration for the model.
    """

    config_class = BalmMoEConfig

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
        super().__init__(config)
        self.alternate_sparsity = self.config.alternate_sparsity
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.embed_dim,
            padding_idx=self.config.padding_idx,
        )

        if self.config.alternate_sparsity:
            layers = []
            for layer_num in range(self.config.num_layers):
                if layer_num % 2 == 0:
                    layers.append(
                        DenseTransformerLayer(
                            embed_dim=self.config.embed_dim,
                            ffn_dim=self.config.ffn_dim,
                            num_heads=self.config.num_heads,
                            max_length=self.config.max_length,
                            dropout=self.config.dropout,
                            attention_dropout=self.config.attention_dropout,
                            token_embedding_dropout=self.config.token_embedding_dropout,
                            layer_norm_eps=self.config.layer_norm_eps,
                            activation=self.config.activation,
                            positional_embedding_type=self.config.positional_embedding_type,
                            pre_norm=self.config.pre_norm,
                        )
                    )
                else:
                    layers.append(
                        SparseTransformerLayer(
                            embed_dim=self.config.embed_dim,
                            ffn_dim=self.config.ffn_dim,
                            num_heads=self.config.num_heads,
                            max_length=self.config.max_length,
                            num_experts=self.config.num_experts,
                            expert_capacity=self.config.expert_capacity,
                            num_shared_experts=self.config.num_shared_experts,
                            top_k=self.config.router_top_k,
                            dropout=self.config.dropout,
                            attention_dropout=self.config.attention_dropout,
                            expert_ffn_dropout=self.config.expert_ffn_dropout,
                            token_embedding_dropout=self.config.token_embedding_dropout,
                            layer_norm_eps=self.config.layer_norm_eps,
                            activation=self.config.activation,
                            positional_embedding_type=self.config.positional_embedding_type,
                            pre_norm=self.config.pre_norm,
                            router_dtype=self.config.router_dtype,
                            router_bias=self.config.router_bias,
                            router_jitter=self.config.router_jitter,
                            router_ignore_padding_tokens=self.config.router_ignore_padding_tokens,
                            expert_choice_router=self.config.expert_choice_router,
                        )
                    )
            self.layers = nn.ModuleList(layers)

        else:
            self.layers = nn.ModuleList(
                [
                    SparseTransformerLayer(
                        embed_dim=self.config.embed_dim,
                        ffn_dim=self.config.ffn_dim,
                        num_heads=self.config.num_heads,
                        max_length=self.config.max_length,
                        num_experts=self.config.num_experts,
                        expert_capacity=self.config.expert_capacity,
                        num_shared_experts=self.config.num_shared_experts,
                        top_k=self.config.router_top_k,
                        dropout=self.config.dropout,
                        attention_dropout=self.config.attention_dropout,
                        expert_ffn_dropout=self.config.expert_ffn_dropout,
                        token_embedding_dropout=self.config.token_embedding_dropout,
                        layer_norm_eps=self.config.layer_norm_eps,
                        activation=self.config.activation,
                        positional_embedding_type=self.config.positional_embedding_type,
                        pre_norm=self.config.pre_norm,
                        router_dtype=self.config.router_dtype,
                        router_bias=self.config.router_bias,
                        router_jitter=self.config.router_jitter,
                        router_ignore_padding_tokens=self.config.router_ignore_padding_tokens,
                        expert_choice_router=self.config.expert_choice_router,
                    )
                    for _ in range(self.config.num_layers)
                ]
            )
        self.embedding_dropout = nn.Dropout(self.config.token_embedding_dropout)
        self.final_norm = nn.LayerNorm(self.config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        output_expert_indexes: bool = False,
        return_dict: bool = True,
    ) -> Union[MaskedLMOutput, tuple]:
        """
        Parameters:
        -----------

        input_ids: torch.LomgTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        output_attentions: bool
            Whether to output attention weights

        output_hidden_states: bool
            Whether to output hidden states

        output_router_logits: bool
            Whether to output router logits

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)


        Returns:
        --------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``MaskedLMOutput`` object:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - router_z_loss (torch.FloatTensor): router z loss
                - router_aux_loss (torch.FloatTensor): router auxiliary loss
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
                - router_logits (torch.FloatTensor): router logits
            If `return_dict` is ``False``, the output is a ``tuple`` with the f0llowing elements:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
                - router_logits (torch.FloatTensor): router logits

        """
        # init
        attn_weights = []
        hidden_states = {}
        router_logits = []
        expert_indexes = []

        # embeddings
        x = self.embed_tokens(x)

        # layers
        for layer_idx, layer in enumerate(self.layers, 1):
            if layer_idx % 2 == 0 or not self.alternate_sparsity:
                # sparse layer, so we need to collect router/expert info
                x = layer(
                    x,
                    attention_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=output_attentions,
                    output_router_logits=output_router_logits,
                )
                if output_attentions:
                    x, attn, router_tuple = x
                    attn_weights.append(attn)
                else:
                    x, router_tuple = x
                router_logits.append(router_tuple[0])
                expert_indexes.append(router_tuple[1])
                if output_hidden_states:
                    hidden_states[layer_idx] = x
            else:
                # dense layer, no router info needed
                x = layer(
                    x,
                    attention_mask=attention_mask,
                    need_weights=output_attentions,
                )
                if output_attentions:
                    x, attn = x
                    attn_weights.append(attn)
                if output_hidden_states:
                    hidden_states[layer_idx] = x

        if self.config.pre_norm:
            x = self.final_norm(x)

        # router losses
        cat_router_logits = torch.cat(router_logits, dim=1)
        cat_expert_indexes = torch.cat(expert_indexes, dim=1)
        router_probs = nn.Softmax(dim=-1)(cat_router_logits)
        z_loss = router_z_loss(cat_router_logits)
        if self.config.expert_choice_router:
            aux_loss = None
        else:
            aux_loss = router_load_balancing_loss(router_probs, cat_expert_indexes)

        # results
        result = MaskedLMOutput(
            last_hidden_state=x,
            router_z_loss=z_loss,
            router_aux_loss=aux_loss,
        )
        if output_attentions:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
        if output_hidden_states:
            result["hidden_states"] = hidden_states
        if output_router_logits:
            result["router_logits"] = cat_router_logits
        if output_expert_indexes:
            result["expert_indexes"] = cat_expert_indexes
        if return_dict:
            return result
        return result.as_tuple()


class BalmMoEForMaskedLM(BalmBase):
    """
    BALM Mixture of Experts model for Masked Language Modeling.

    Parameters
    ----------
    config: BalmMoEConfig
        Configuration for the model.
    """

    config_class = BalmMoEConfig

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
        super().__init__(config)
        self.balm = BalmMoEModel(
            config=self.config,
        )
        self.lm_head = BalmLMHead(
            embed_dim=self.config.embed_dim,
            output_dim=self.config.vocab_size,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.router_z_loss_coef = self.config.router_z_loss_coef
        self.router_aux_loss_coef = self.config.router_aux_loss_coef

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = True,
        output_expert_indexes: bool = False,
        return_dict: bool = True,
    ) -> Union[dict, tuple]:
        """
        Forward pass

        Parameters
        ----------
        input_ids: torch.LongTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        key_padding_mask: torch.BoolTensor
            Key padding mask

        labels: torch.LongTensor
            Labels

        output_attentions: bool
            Whether to output attention weights

        output_hidden_states: bool
            Whether to output hidden states

        output_router_logits: bool
            Whether to output router logits

        output_expert_indexes: bool
            Whether to output expert indexes

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)
        """
        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_expert_indexes=output_expert_indexes,
            return_dict=True,
        )
        x = outputs["last_hidden_state"]
        router_z_loss = outputs["router_z_loss"]
        router_aux_loss = outputs["router_aux_loss"]

        # LM head
        lm_logits = self.lm_head(x)
        outputs["logits"] = lm_logits

        # loss
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss = self.criterion(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
            outputs["lm_loss"] = loss

            if output_router_logits:
                z_loss = self.router_z_loss_coef * (router_z_loss)
                outputs["router_z_loss"] = z_loss
                if self.config.expert_choice_router:
                    loss = loss + z_loss
                else:
                    aux_loss = self.router_aux_loss_coef * (router_aux_loss)
                    outputs["router_aux_loss"] = aux_loss
                    loss = loss + z_loss + aux_loss
            outputs["loss"] = loss

        # outputs
        if return_dict:
            return outputs.as_dict()
        return outputs.as_tuple()


class BalmMoEForSequenceClassification(BalmBase):
    """
    BALM model for sequence classification. Uses the dense BALM transformer model and adds
    a sequence-level classification head.

    Parameters
    ----------
    config : BalmConfig
        The configuration object defining model architecture and hyperparameters.
    """

    config_class = BalmMoEConfig

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
        super().__init__(config)
        # model
        self.balm = BalmMoEModel(config=self.config)

        # classifier
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.dropout
        )
        classifier_activation = (
            self.config.classifier_activation
            if self.config.classifier_activation is not None
            else "tanh"
        )
        # classifier_dropout = self.config.dropout
        # classifier_activation = "tanh"
        self.classifier = BalmClassificationHead(
            embed_dim=self.config.embed_dim,
            num_labels=self.config.num_labels,
            dropout=classifier_dropout,
            activation=classifier_activation,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> MaskedLMOutput:
        """
        Parameters
        ----------

        x : torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            The output tensor. The shape is (batch_size, seq_len, vocab_size).
        """
        x = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=output_attentions,
        )
        if output_attentions:
            x, attn = x
        logits = self.classifier(x)

        classifier_loss = None
        if labels is not None:
            classifier_loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        output = ClassifierOutput(
            logits=logits,
            loss=classifier_loss,
        )
        if output_attentions:
            output.attentions = attn
        if output_hidden_states:
            output.hidden_states = x
        if return_dict:
            return output.as_dict()
        return output.as_tuple()

"""Local G-SWaN (Graph Self-attention WaveNet) implementation.

Independent rewrite of Sections 3.2.2-3.2.6 (Eqs. 4-8) of Prabowo et al.,
IoTDI 2023 (arXiv 2302.09956), with omissions resolved by reading the official
repository (``aprbw/G-SWaN``, revision ``d95c2af``). Nothing was copied.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import to_spatiotemporal

TimeFeature = Literal["time_of_week", "time_of_day"]

# Large negative score for non-edges before the attention softmax (finite so a
# node without any edge degrades to a uniform softmax instead of NaN).
_NON_EDGE = -9.0e15


def wavenet_lengths(seq_len: int, kernel_size: int, blocks: int, layers: int) -> tuple[int, list[int], list[int]]:
    """Return ``(padded_len, dilations, lengths)`` of the valid dilated stack.

    The input is left-padded to the receptive field when shorter; each layer
    shortens the time axis by ``dilation * (kernel_size - 1)``; ``lengths[i]``
    is the time length after layer ``i``.
    """
    dilations = [2**layer for _ in range(blocks) for layer in range(layers)]
    receptive = 1 + sum(d * (kernel_size - 1) for d in dilations)
    padded = max(seq_len, receptive)
    lengths, current = [], padded
    for dilation in dilations:
        current -= dilation * (kernel_size - 1)
        lengths.append(current)
    return padded, dilations, lengths


def pair_attention(
    keys: torch.Tensor,
    queries: torch.Tensor,
    support: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Eq. (6): input-adaptive attention over the edges of one support.

    ``keys``/``queries`` are ``[B, E, N]``; the score ``e[v, w] = sum_e K[e, v] Q[e, w]``
    passes through Mish (sigma), non-edges (``support <= 0``) are masked, the
    scores are divided by the temperature and normalized over the source node
    ``v`` (so ``x' = x @ alpha`` aggregates sources into every target ``w``).
    """
    scores = F.mish(torch.einsum("bev,bew->bvw", keys, queries))
    scores = torch.where(support.unsqueeze(0) > 0, scores, torch.full_like(scores, _NON_EDGE))
    return torch.softmax(scores / temperature, dim=1)


class SpatialGraphTransformer(nn.Module):
    """SGT (Eqs. 4-8): diffusion over static and attention-reweighted supports.

    For every support ``A`` there is one plain diffusion head (``A`` itself) and
    ``num_heads`` attention heads ``alpha(A | x, e1, e2)``; each head contributes
    hops ``1..order``. The input and all hop terms are concatenated on the
    channel axis and projected by a 1x1 convolution (Eq. 4's sum over ``k, h``
    with per-term weights), followed by dropout.
    """

    def __init__(
        self,
        channels: int,
        time_len: int,
        support_len: int,
        num_heads: int,
        order: int,
        dropout: float,
        softmax_temp: float,
    ) -> None:
        super().__init__()
        self.support_len, self.num_heads, self.order = support_len, num_heads, order
        self.dropout, self.softmax_temp = dropout, softmax_temp
        # FC_K / FC_Q (Eqs. 7-8): one kernel spanning the whole remaining time axis,
        # 2 * channels outputs per node; one pair per head, shared across supports.
        self.key = nn.ModuleList(nn.Conv2d(channels, 2 * channels, (1, time_len)) for _ in range(num_heads))
        self.query = nn.ModuleList(nn.Conv2d(channels, 2 * channels, (1, time_len)) for _ in range(num_heads))
        self.terms = (num_heads + 1) * order * support_len + 1
        self.project = nn.Conv2d(self.terms * channels, channels, 1)

    def head_supports(
        self, x: torch.Tensor, support: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return ``[A, alpha_1(A), ..., alpha_H(A)]`` each broadcast to ``[B, N, N]``."""
        batch, channels, nodes, _ = x.shape
        key_input = x + e1.t().reshape(1, channels, nodes, 1)  # Eq. (7): x + e1
        query_input = x + e2.reshape(1, channels, nodes, 1)  # Eq. (8): x + e2
        heads = [support.expand(batch, nodes, nodes)]
        for key, query in zip(self.key, self.query):
            alpha = pair_attention(
                key(key_input).squeeze(-1), query(query_input).squeeze(-1), support, self.softmax_temp
            )
            heads.append(F.dropout(alpha, self.dropout, training=self.training))
        return heads

    def forward(
        self, x: torch.Tensor, supports: list[torch.Tensor], e1: torch.Tensor, e2: torch.Tensor
    ) -> torch.Tensor:
        if len(supports) != self.support_len:
            raise ValueError("support count does not match configured support_len")
        terms = [x]
        for support in supports:
            for alpha in self.head_supports(x, support, e1, e2):
                hop = x
                for _ in range(self.order):
                    hop = torch.einsum("bcvl,bvw->bcwl", hop, alpha)
                    terms.append(hop)
        out = self.project(torch.cat(terms, dim=1))
        return F.dropout(out, self.dropout, training=self.training)


class GSWaNLayer(nn.Module):
    """One G-SWaN layer (Fig. 5): gated WaveNet, SGT, residuals, BatchNorm."""

    def __init__(
        self,
        channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        time_len: int,
        support_len: int,
        num_heads: int,
        order: int,
        dropout: float,
        softmax_temp: float,
        spatial: bool,
    ) -> None:
        super().__init__()
        self.filter = nn.Conv2d(channels, channels, (1, kernel_size), dilation=(1, dilation))
        self.gate = nn.Conv2d(channels, channels, (1, kernel_size), dilation=(1, dilation))
        self.skip = nn.Conv2d(channels, skip_channels, 1)
        self.spatial = (
            SpatialGraphTransformer(channels, time_len, support_len, num_heads, order, dropout, softmax_temp)
            if spatial
            else None
        )
        self.norm = nn.BatchNorm2d(channels) if spatial else None

    def forward(
        self, x: torch.Tensor, supports: list[torch.Tensor], e1: torch.Tensor, e2: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # Valid (unpadded) dilated convolution: time shrinks by dilation * (kernel - 1).
        temporal = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        skip = self.skip(temporal)[..., -1:]
        if self.spatial is None or self.norm is None:
            return None, skip
        # Residual around the SGT, then the layer residual from the input, then BatchNorm.
        mixed = temporal + self.spatial(temporal, supports, e1, e2)
        hidden = mixed + x[..., -mixed.shape[-1] :]
        return self.norm(hidden), skip


def _static_supports(adj_mx: np.ndarray | None, num_nodes: int) -> list[torch.Tensor]:
    # Predefined road graph A_r as forward/backward random-walk supports; omitted
    # when no adjacency is supplied (the adaptive support A_adp is always present).
    if adj_mx is None:
        return []
    adjacency = np.asarray(adj_mx, dtype=np.float32)
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("adj_mx shape must match num_nodes")
    return adj_to_supports(adjacency, "doubletransition")


class Model(nn.Module):
    """G-SWaN: Graph WaveNet backbone whose graph convolutions are SGT modules."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        hidden_channels: int = 40,
        skip_channels: int = 320,
        end_channels: int = 640,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        num_heads: int = 3,
        order: int = 2,
        softmax_temp: float = 7.0,
        dropout: float = 0.3,
        time_feature: TimeFeature = "time_of_week",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, num_nodes) < 1:
            raise ValueError("GSWaN sizes must be positive")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.time_feature = time_feature
        self.padded_len, dilations, lengths = wavenet_lengths(seq_len, kernel_size, blocks, layers)
        static = _static_supports(adj_mx, num_nodes)
        self.num_static = len(static)
        for index, support in enumerate(static):
            self.register_buffer(f"static_support_{index}", support)
        # Source/target node embeddings e1 [N, D] and e2 [D, N]; their width equals the
        # hidden width because Eqs. 7-8 add them to the latent input.
        self.e1 = nn.Parameter(torch.randn(num_nodes, hidden_channels))
        self.e2 = nn.Parameter(torch.randn(hidden_channels, num_nodes))
        # Initial embedding (Sec. 3.2.2): FC on the metric plus an activated FC on time.
        self.metric_embedding = nn.Conv2d(1, hidden_channels, 1)
        self.time_embedding = nn.Conv2d(1, hidden_channels, 1)
        support_len = self.num_static + 1
        last = len(dilations) - 1
        self.layers = nn.ModuleList(
            GSWaNLayer(
                hidden_channels, skip_channels, kernel_size, dilation, length, support_len,
                num_heads, order, dropout, softmax_temp, spatial=index < last,
            )
            for index, (dilation, length) in enumerate(zip(dilations, lengths))
        )
        self.end_hidden = nn.Conv2d(skip_channels, end_channels, 1)
        self.end_output = nn.Conv2d(end_channels, pred_len, 1)

    def supports(self) -> list[torch.Tensor]:
        """``[A_r forward, A_r backward, A_adp]`` with Eq. (5) for the adaptive one."""
        static = [getattr(self, f"static_support_{index}") for index in range(self.num_static)]
        return static + [adaptive_node_embedding_adjacency(self.e1, self.e2)]

    def inputs(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return metric and time channels, each ``[B, 1, N, T]``."""
        data = to_spatiotemporal(x_enc, x_mark_enc)  # [B, T, N, 1 + F]
        metric = data[..., 0]
        if x_mark_enc is not None and x_mark_enc.dim() == 4:
            time = data[..., 1]  # node-structured covariates: first covariate channel
        elif self.time_feature == "time_of_day":
            time = data[..., 1]
        else:
            # Fraction of the week elapsed: (weekday + time_in_day) / 7.
            time = data[..., 2] + data[..., 1] / 7.0
        return metric.transpose(1, 2).unsqueeze(1), time.transpose(1, 2).unsqueeze(1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or tuple(x_enc.shape[1:]) != (self.seq_len, self.num_nodes):
            raise ValueError(f"GSWaN expects (B, {self.seq_len}, {self.num_nodes}) values")
        metric, time = self.inputs(x_enc, x_mark_enc)
        pad = self.padded_len - self.seq_len
        metric, time = F.pad(metric, (pad, 0)), F.pad(time, (pad, 0))
        x = self.metric_embedding(metric) + F.mish(self.time_embedding(time))
        supports = self.supports()
        skip = None
        for layer in self.layers:
            hidden, s = layer(x, supports, self.e1, self.e2)
            skip = s if skip is None else skip + s
            if hidden is not None:
                x = hidden
        out = self.end_output(F.mish(self.end_hidden(F.mish(skip))))  # [B, pred_len, N, 1]
        return out.squeeze(-1)


__all__ = ["GSWaNLayer", "Model", "SpatialGraphTransformer", "pair_attention", "wavenet_lengths"]

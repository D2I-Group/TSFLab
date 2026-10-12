"""Local Esiformer: interpolated inverted Transformer with a sparse FFN.

Paper map (Guo et al., arXiv 2410.05726, Section II):
``X'_{2L-1} = Interpolation(X_L)``, ``H = Embedding(X')``,
``H' = TransformerBlock(H)`` with the FFN replaced by a sparse FFN,
``Y_T = Projection(H')``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.marks import adapt_tslib_marks
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder
from tsflab.models._inputs import check_window


def interpolated_length(seq_len: int, method: str) -> int:
    """Token length after interpolation: ``2L - 1`` (TwoAver) or ``3L/2 - 1`` (FourAver)."""
    if method == "two_aver":
        return 2 * seq_len - 1
    if method == "four_aver":
        return 3 * seq_len // 2 - 1
    return seq_len


def two_aver(values: torch.Tensor) -> torch.Tensor:
    """TwoAver along time of ``[B, L, C]`` -> ``[B, 2L - 1, C]``.

    ``x'_{2i-1} = x_i`` and ``x'_{2i} = (x_i + x_{i+1}) / 2``: every original
    step is kept and the midpoint of each neighbouring pair is inserted.
    """
    batch, length, channels = values.shape
    midpoints = 0.5 * (values[:, :-1] + values[:, 1:])
    pairs = torch.stack((values[:, :-1], midpoints), dim=2).reshape(batch, 2 * (length - 1), channels)
    return torch.cat((pairs, values[:, -1:]), dim=1)


def four_aver(values: torch.Tensor) -> torch.Tensor:
    """FourAver along time of ``[B, L, C]`` (even ``L``) -> ``[B, 3L/2 - 1, C]``.

    ``x'_{3i-2} = x_{2i-1}``, ``x'_{3i-1} = x_{2i}`` and
    ``x'_{3i} = (x_{2i-1} + x_{2i} + x_{2i+1} + x_{2i+2}) / 4``; the final pair
    has no right neighbours and receives no inserted value.
    """
    batch, length, channels = values.shape
    if length % 2 or length < 4:
        raise ValueError("four_aver interpolation needs an even length of at least 4")
    pairs = values.reshape(batch, length // 2, 2, channels)
    windows = 0.25 * (pairs[:, :-1].sum(dim=2) + pairs[:, 1:].sum(dim=2))
    triples = torch.cat((pairs[:, :-1], windows.unsqueeze(2)), dim=2).reshape(batch, -1, channels)
    return torch.cat((triples, pairs[:, -1]), dim=1)


def interpolate(values: torch.Tensor, method: str) -> torch.Tensor:
    if method == "two_aver":
        return two_aver(values)
    if method == "four_aver":
        return four_aver(values)
    return values


def interpolate_marks(marks: torch.Tensor, method: str) -> torch.Tensor:
    """Calendar features follow the values with midpoints at inserted steps.

    For FourAver the inserted value sits between ``x_{2i}`` and ``x_{2i+1}``
    (1-indexed), so its timestamp is their midpoint: the TwoAver marks with
    every fourth midpoint (positions ``1, 5, 9, ...``) removed.
    """
    if method == "four_aver":
        doubled = two_aver(marks)
        keep = [index for index in range(doubled.shape[1]) if (index - 1) % 4 != 0]
        return doubled[:, keep]
    return interpolate(marks, method)


class SparseLinear(nn.Module):
    """Linear map with a fixed random connectivity mask (Sparse FFN, Section II-C).

    ``round((1 - sparsity) * in * out)`` connections are drawn uniformly without
    replacement at construction and never change; weights and bias are
    initialised from ``U(-1/sqrt(in), 1/sqrt(in))`` and pruned entries are
    held at zero, so the layer is ``y = (W * M) x + b``.
    """

    def __init__(self, in_features: int, out_features: int, sparsity: float) -> None:
        super().__init__()
        total = in_features * out_features
        kept = round((1.0 - sparsity) * total)
        self.in_features, self.out_features, self.sparsity = in_features, out_features, sparsity
        mask = torch.zeros(total)
        mask[torch.randperm(total)[:kept]] = 1.0
        self.register_buffer("mask", mask.view(out_features, in_features))
        bound = 1.0 / math.sqrt(in_features)
        weight = torch.empty(out_features, in_features).uniform_(-bound, bound) * self.mask
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight * self.mask, self.bias)


class SparseEncoderLayer(nn.Module):
    """Post-norm encoder block: attention, then the sparse (or dense) FFN.

    ``x = Norm(x + Drop(Attn(x)))``; ``out = Norm(x + Drop(W2 Drop(act(W1 x))))``
    with sparse ``W1: d_model -> d_ff`` and ``W2: d_ff -> d_model``.
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str,
        sparse_ffn: bool,
        sparsity: float,
    ) -> None:
        super().__init__()
        self.attention = attention
        if sparse_ffn:
            self.ffn_in: nn.Module = SparseLinear(d_model, d_ff, sparsity)
            self.ffn_out: nn.Module = SparseLinear(d_ff, d_model, sparsity)
        else:
            self.ffn_in = nn.Linear(d_model, d_ff)
            self.ffn_out = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def feed_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.ffn_in(tokens)))
        return self.dropout(self.ffn_out(hidden))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attended, attention = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.feed_forward(x)), attention


def _check_interpolation(interpolation: str, seq_len: int) -> None:
    if interpolation == "four_aver" and (seq_len % 2 or seq_len < 4):
        raise ValueError("four_aver interpolation needs an even seq_len of at least 4")
    if interpolation == "two_aver" and seq_len < 2:
        raise ValueError("two_aver interpolation needs seq_len >= 2")


class Model(nn.Module):
    """Esiformer: interpolation, variate tokens, sparse-FFN encoder, linear head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        interpolation: str = "two_aver",
        sparse_ffn: bool = True,
        sparsity: float = 0.3,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
        use_marks: bool = True,
        freq: str = "h",
    ) -> None:
        super().__init__()
        _check_interpolation(interpolation, seq_len)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.interpolation = interpolation
        self.token_length = interpolated_length(seq_len, interpolation)
        self.use_norm, self.use_marks, self.freq = use_norm, use_marks, freq
        self.embedding = DataEmbedding_inverted(self.token_length, d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                SparseEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                    sparse_ffn,
                    sparsity,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, pred_len)

    def calendar_tokens(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        """TSLib ``timeF`` calendar features, interpolated like the values."""
        if not self.use_marks or x_mark_enc is None:
            return None
        if x_mark_enc.ndim != 3 or x_mark_enc.shape[1] != self.seq_len:
            raise ValueError("x_mark_enc must be [batch, seq_len, mark_columns]")
        marks = adapt_tslib_marks(x_mark_enc, embed_type="timeF", freq=self.freq)
        return interpolate_marks(marks, self.interpolation)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        # The official pipeline interpolates the raw window, then normalizes it.
        values = interpolate(x_enc, self.interpolation)
        if self.use_norm:
            values, mean, stdev = standardize(values)
        marks = self.calendar_tokens(x_mark_enc)
        if marks is not None:
            marks = marks.to(dtype=values.dtype)
        tokens = self.embedding(values, marks)  # [B, C (+ mark features), d_model]
        encoded, _ = self.encoder(tokens, attn_mask=None)
        forecast = self.projector(encoded).transpose(1, 2)[:, :, : self.enc_in]
        if self.use_norm:
            forecast = destandardize(forecast, mean, stdev)
        return forecast

"""Paper-driven local implementation of TSRM (and its TSRM_IFC variant).

TSRM (Leppich et al., arXiv 2504.18878) processes every channel of the
RevIN-normalized window independently with a shared backbone (Sec. 2.1). Each
scalar is lifted to ``d`` features, then ``N`` encoding layers (ELs) run in
sequence. An EL's representation layer (RL, Eq. 1) applies ``K`` strided,
optionally dilated 1D convolutions (stride = kernel size, no padding) and
concatenates their outputs along the sequence axis; two pre-activation residual
blocks (normalization, GELU, multi-head self-attention, dropout; then
normalization, GELU, feed-forward, dropout) refine that representation; a merge
layer (ML, Eq. 2) splits it back, inverts each convolution with a transposed
convolution to length ``T``, and fuses the ``K`` results with a linear map. A
global residual accumulates the representations across ELs. A linear head maps
the flattened ``[T, d]`` output to the horizon. TSRM_IFC widens the feed-forward
to all ``F x d`` channel features of a position. Where the paper is less
specific than the pinned official code, this module follows the code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.revin import RevIN


def entmax15(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1.5-entmax (Peters et al., 2019): ``p_i = [z_i / 2 - tau]_+^2`` with ``sum_i p_i = 1``.

    ``tau`` is found exactly by sorting: for the ``k`` largest halved scores the
    threshold solving ``sum_{i<=k} (x_i - tau)^2 = 1`` is
    ``mean_k - sqrt((1 - k (meansq_k - mean_k^2)) / k)``; the support is the
    largest ``k`` with ``tau_k <= x_(k)``.
    """
    x = scores.transpose(dim, -1)
    x = (x - x.max(dim=-1, keepdim=True).values) / 2
    ordered = x.sort(dim=-1, descending=True).values
    k = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    mean = ordered.cumsum(-1) / k
    mean_sq = ordered.square().cumsum(-1) / k
    delta = (1 - k * (mean_sq - mean.square())) / k
    tau = mean - delta.clamp(min=0).sqrt()
    support = (tau <= ordered).sum(dim=-1, keepdim=True)
    tau_star = tau.gather(-1, support - 1)
    return (x - tau_star).clamp(min=0).square().transpose(dim, -1)


def conv_length(length: int, kernel: int, dilation: int) -> int:
    """Eq. (1): ``D_j = floor((T - delta_j (s_j - 1) - 1) / s_j) + 1`` with stride ``s_j``."""
    return (length - dilation * (kernel - 1) - 1) // kernel + 1


class SelfAttention(nn.Module):
    """Multi-head self-attention over the representation sequence (softmax or 1.5-entmax)."""

    def __init__(self, d_model: int, n_heads: int, attention: str, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attention = attention
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def weights(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        return entmax15(scores) if self.attention == "entmax15" else torch.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        mixed = self.dropout(self.weights(q, k)) @ v
        return self.out(mixed.transpose(1, 2).reshape(batch, length, width))


class FeedForward(nn.Module):
    """Position-wise block of the official code: ``ReLU(W_2 (W_1 x))``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.second(self.first(x)))


class EncodingLayer(nn.Module):
    """One EL: RL (Eq. 1) -> attention block -> feed-forward block -> ML (Eq. 2)."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        conv_layers: tuple[tuple[int, int, int], ...],
        attention: str,
        dropout: float,
        enc_in: int,
        inter_feature: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.inter_feature = inter_feature
        self.representations = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.lengths: list[int] = []
        for kernel, dilation, groups in conv_layers:
            groups = d_model if groups == -1 else groups
            length = conv_length(seq_len, kernel, dilation)
            restored = (length - 1) * kernel + dilation * (kernel - 1) + 1
            self.lengths.append(length)
            self.representations.append(
                nn.Conv1d(d_model, d_model, kernel, stride=kernel, dilation=dilation, groups=groups)
            )
            self.merges.append(
                nn.ConvTranspose1d(
                    d_model, d_model, kernel, stride=kernel, dilation=dilation, groups=groups,
                    output_padding=seq_len - restored,
                )
            )
        self.norm_attention = nn.GroupNorm(1, d_model)
        self.attention = SelfAttention(d_model, n_heads, attention, dropout)
        self.norm_ffn = nn.GroupNorm(1, d_model)
        self.feed_forward = FeedForward(d_model * (enc_in if inter_feature else 1))
        self.dropout = nn.Dropout(dropout)
        self.fuse = nn.Linear(d_model * len(conv_layers), d_model)

    @staticmethod
    def _norm(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
        """GroupNorm with one group: normalize each sequence over all positions and features."""
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (1): ``[B', T, d] -> [B', D, d]``, ELU after every convolution."""
        return torch.cat([F.elu(conv(x.transpose(1, 2))) for conv in self.representations], dim=-1).transpose(1, 2)

    def merge(self, r: torch.Tensor) -> torch.Tensor:
        """Eq. (2): split ``[B', D, d]``, transposed convolutions to ``T``, concat, Linear(K d -> d)."""
        parts = r.split(self.lengths, dim=1)
        restored = [merge(part.transpose(1, 2)).transpose(1, 2) for merge, part in zip(self.merges, parts)]
        return self.fuse(torch.cat(restored, dim=-1))

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inter_feature:
            return self.feed_forward(x)
        # TSRM_IFC: one feed-forward over the F x d features of each position.
        rows, length, width = x.shape
        joint = x.reshape(rows // self.enc_in, self.enc_in, length, width).permute(0, 2, 1, 3)
        joint = self.feed_forward(joint.reshape(rows // self.enc_in, length, self.enc_in * width))
        joint = joint.reshape(rows // self.enc_in, length, self.enc_in, width).permute(0, 2, 1, 3)
        return joint.reshape(rows, length, width)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.represent(x)
        if residual is None:
            residual = r
        else:
            r = r + residual
        h = r + self.dropout(self.attention(F.gelu(self._norm(self.norm_attention, r))))
        h = h + self.dropout(self._feed_forward(F.gelu(self._norm(self.norm_ffn, h))))
        return self.merge(h), residual + h


def _checked_conv_layers(conv_layers, seq_len: int) -> tuple[tuple[int, int, int], ...]:
    conv_layers = tuple(tuple(int(v) for v in layer) for layer in conv_layers)
    for kernel, dilation, _ in conv_layers:
        if conv_length(seq_len, kernel, dilation) < 1:
            raise ValueError("every convolution must fit the input window")
    return conv_layers


class Model(nn.Module):
    """TSRM forecaster with the four-input TSFLab interface."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 16,
        e_layers: int = 7,
        conv_layers: tuple[tuple[int, int, int], ...] = ((3, 1, -1), (5, 2, -1), (10, 3, -1)),
        attention: str = "entmax15",
        dropout: float = 0.25,
        inter_feature: bool = False,
    ) -> None:
        super().__init__()
        conv_layers = _checked_conv_layers(conv_layers, seq_len)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, affine=True)
        # Position-wise lift of each scalar to d features, shared by all channels.
        self.value_embedding = nn.Linear(1, d_model, bias=False)
        self.layers = nn.ModuleList(
            EncodingLayer(seq_len, d_model, n_heads, conv_layers, attention, dropout, enc_in, inter_feature)
            for _ in range(e_layers)
        )
        self.head = FlattenForecastHead(False, enc_in, seq_len * d_model, pred_len)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, F]`` (normalized) -> ``[B * F, T, d]`` after all ELs."""
        batch, length, channels = x.shape
        z = self.value_embedding(x.transpose(1, 2).reshape(batch * channels, length, 1))
        residual = None
        for layer in self.layers:
            z, residual = layer(z, residual)
        return z

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.revin(x_enc, "norm")
        batch, length, channels = x.shape
        z = self.encode(x).reshape(batch, channels, length, -1)
        forecast = self.head(z).transpose(1, 2)  # flatten [T, d] -> horizon, per channel
        return self.revin(forecast, "denorm")

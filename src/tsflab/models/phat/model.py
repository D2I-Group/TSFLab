"""Clean-room PHAT implementation from Ma et al. (ICLR 2026).

The implementation follows the public equations rather than the incomplete
author repository: FFT-derived periods form phase-by-cycle buckets; equations
(4)-(12) define positive-negative X-shaped attention; the bucket is unfolded
and projected to the forecast horizon. Variates whose detected period is not
shorter than the lookback form the paper's zero bucket: they are not folded,
and both attentions use absolute instead of periodic distance (Sec. 3.2,
App. B). Dataset-specific training recipes are not reproduced.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN


def _distance_masks(period: int, device: torch.device, periodic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Eq. (7)-(8) positive/negative modulation sets (absolute distance in the zero bucket)."""
    index = torch.arange(period, device=device)
    difference = index[:, None] - index[None, :]
    if periodic:
        distance = torch.minimum(difference.remainder(period), (-difference).remainder(period))
    else:
        distance = difference.abs()
    query_to_source = distance[:, None, :]
    query_to_key = distance[:, :, None]
    self_position = torch.eye(period, device=device, dtype=torch.bool)[:, None, :]
    return (
        ((query_to_source < query_to_key) | self_position).to(torch.float32),
        ((query_to_source > query_to_key) | self_position).to(torch.float32),
    )


class PositiveNegativeAttention(nn.Module):
    """Paper Eq. (4)-(12) on ``[batch, phase, cycle, width]`` buckets."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.scale = self.head_width**-0.5
        self.query = nn.Linear(width, 2 * width)
        self.key = nn.Linear(width, 2 * width)
        self.value = nn.Linear(width, width)
        self.gate = nn.Linear(width, heads)
        self.aligned_scale = nn.Parameter(torch.tensor(self.scale))
        self.alpha = nn.Parameter(torch.ones(heads))
        self.gamma = nn.Parameter(torch.ones(heads, self.head_width))
        self.beta = nn.Parameter(torch.zeros(heads, self.head_width))
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, phase, cycles, _ = tensor.shape
        return tensor.reshape(batch, phase, cycles, self.heads, self.head_width).permute(0, 3, 1, 2, 4)

    def forward(self, bucket: torch.Tensor, periodic: bool = True) -> torch.Tensor:
        q_positive, q_negative = self.query(bucket).chunk(2, dim=-1)
        k_positive, k_negative = self.key(bucket).chunk(2, dim=-1)
        q_positive, q_negative = self._heads(q_positive), self._heads(q_negative)
        k_positive, k_negative = self._heads(k_positive), self._heads(k_negative)
        value = self._heads(self.value(bucket))
        strength = torch.sigmoid(self.gate(bucket)).permute(0, 3, 1, 2)

        positive_logits = torch.einsum("bhpnd,bhqnd->bhpnq", q_positive, k_positive) * self.scale
        negative_logits = torch.einsum("bhpnd,bhqnd->bhpnq", q_negative, k_negative) * self.scale
        positive_set, negative_set = _distance_masks(bucket.shape[1], bucket.device, periodic)
        positive_logits = positive_logits - torch.einsum(
            "pqs,bhpns->bhpnq", positive_set.to(bucket.dtype), F.softplus(positive_logits)
        )
        negative_logits = negative_logits - torch.einsum(
            "pqs,bhpns->bhpnq", negative_set.to(bucket.dtype), F.softplus(negative_logits)
        )
        offset_attention = torch.softmax(positive_logits, dim=-1)
        offset_attention = offset_attention - strength.unsqueeze(-1) * torch.softmax(negative_logits, dim=-1)

        aligned_attention = torch.softmax(
            torch.einsum("bhpnd,bhpmd->bhpnm", q_positive, k_positive) * self.aligned_scale,
            dim=-1,
        )
        aligned_value = torch.einsum("bhpnm,bhpmd->bhpnd", aligned_attention, value)
        attended = torch.einsum("bhpnq,bhqnd->bhpnd", offset_attention, aligned_value)
        attended = attended + strength.unsqueeze(-1) * value
        shaped = self.gamma[None, :, None, None, :] * torch.tanh(
            self.alpha[None, :, None, None, None] * attended
        ) + self.beta[None, :, None, None, :]
        merged = shaped.permute(0, 2, 3, 1, 4).flatten(-2)
        return self.dropout(self.output(merged))


class _PHATBlock(nn.Module):
    def __init__(self, width: int, heads: int, attention_dropout: float, ffn_dropout: float, expansion: float) -> None:
        super().__init__()
        hidden = max(width, round(width * expansion))
        self.attention_norm = nn.LayerNorm(width)
        self.attention = PositiveNegativeAttention(width, heads, attention_dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_gate = nn.Linear(width, 2 * hidden)
        self.ffn_output = nn.Linear(hidden, width)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x: torch.Tensor, periodic: bool = True) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), periodic)
        gate, value = self.ffn_gate(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.ffn_output(self.ffn_dropout(F.silu(gate) * value))


class Model(nn.Module):
    """Period Heterogeneity-Aware Transformer forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 8,
        d_layers: int = 1,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        ffn_expand_ratio: float = 2.66667,
        period_topk: int = 1,
        period_list: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_topk = period_topk
        self.period_list = tuple(period_list) if period_list else None
        self.normalization = RevIN(enc_in)
        self.base_projection = nn.Linear(seq_len, pred_len)
        self.bucket_embedding = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            _PHATBlock(d_model, n_heads, attn_dropout, ffn_dropout, ffn_expand_ratio)
            for _ in range(d_layers)
        )
        self.bucket_output = nn.Linear(d_model, 1)
        self.forecast_projection = nn.Linear(seq_len, pred_len)

    def _periods(self, series: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        if self.period_list is not None:
            periods = [min(self.seq_len, value) for value in self.period_list[: self.period_topk]]
            return periods, series.new_full((len(periods),), 1.0 / len(periods))
        amplitude = torch.fft.rfft(series, dim=1).abs().mean(dim=0)
        amplitude[0] = 0
        count = min(self.period_topk, max(1, amplitude.numel() - 1))
        values, frequencies = torch.topk(amplitude, count)
        periods = (self.seq_len / frequencies.clamp_min(1)).round().long().clamp(1, self.seq_len)
        return periods.tolist(), torch.softmax(values, dim=0)

    def _bucket_path(self, series: torch.Tensor, period: int) -> torch.Tensor:
        embedded = self.bucket_embedding(series.unsqueeze(-1))
        padding = (-self.seq_len) % period
        if padding:
            embedded = F.pad(embedded, (0, 0, 0, padding))
        cycles = embedded.shape[1] // period
        bucket = embedded.reshape(embedded.shape[0], cycles, period, -1).transpose(1, 2)
        periodic = period < self.seq_len  # otherwise the zero bucket: no periodicity, absolute distance
        for block in self.blocks:
            bucket = block(bucket, periodic)
        unfolded = bucket.transpose(1, 2).reshape(embedded.shape[0], cycles * period, -1)
        return self.bucket_output(unfolded[:, : self.seq_len]).squeeze(-1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"expected [batch, {self.seq_len}, channels], got {tuple(x_enc.shape)}")
        normalized = self.normalization(x_enc, "norm")
        base = self.base_projection(normalized.transpose(1, 2)).transpose(1, 2)
        refinements = []
        for channel in range(normalized.shape[-1]):
            series = normalized[:, :, channel]
            periods, weights = self._periods(series)
            paths = torch.stack([self._bucket_path(series, period) for period in periods], dim=0)
            combined = torch.einsum("k,kbl->bl", weights.to(paths.dtype), paths)
            refinements.append(self.forecast_projection(combined))
        refined = torch.stack(refinements, dim=-1)
        output = 0.5 * (base + refined)
        return self.normalization(output, "denorm")

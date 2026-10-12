"""Clean-room CauAir based on the IJCAI 2025 CachLormer equations.

Historical AQI and meteorological covariates are summarized independently,
then coupled by a Cache-based Lightweight Transformer.  Cache-attention assigns
fine-grained stations to a small learnable cache, aggregates values in that
coarse view, and broadcasts the result back to stations in O(P*N*d).  A second
CachLormer propagates the learned past association through future covariates.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import (
    coerce_time_length,
    future_time_features,
    to_spatiotemporal,
)


class CacheAttention(nn.Module):
    """Paper cache-attention (Eqs. 6--11) with multi-head coarse regions."""

    def __init__(self, d_model: int, heads: int, cache_count: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.cache = nn.Parameter(torch.randn(heads, cache_count, self.head_dim) * 0.02)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, nodes, width = values.shape
        query = self.query(values).reshape(batch, nodes, self.heads, self.head_dim)
        value = self.value(values).reshape(batch, nodes, self.heads, self.head_dim)
        assignment = torch.einsum("bnhd,hpd->bhnp", query, self.cache)
        assignment = (assignment / math.sqrt(self.head_dim)).softmax(-1)
        normalizer = assignment.sum(2, keepdim=True).transpose(2, 3).clamp_min(1e-6)
        coarse = torch.einsum("bhnp,bnhd->bhpd", assignment, value) / normalizer
        restored = torch.einsum("bhnp,bhpd->bnhd", assignment, coarse)
        return self.output(restored.reshape(batch, nodes, width))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, hidden * 2)
        self.output = nn.Linear(hidden, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        signal, gate = self.input(values).chunk(2, dim=-1)
        return self.output(signal * F.silu(gate))


class CachLormer(nn.Module):
    """Paper Eq. (5): weighted parallel cache-attention and SwiGLU FFN."""

    def __init__(self, d_model: int, heads: int, cache_count: int) -> None:
        super().__init__()
        self.attention = CacheAttention(d_model, heads, cache_count)
        self.feed_forward = SwiGLU(d_model, d_model * 2)
        self.mixture_logits = nn.Parameter(torch.zeros(2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mixture = torch.softmax(self.mixture_logits, dim=0)
        return mixture[0] * self.attention(values) + mixture[1] * self.feed_forward(values)


class Model(nn.Module):
    """Causal covariate forecaster with two-stage CachLormer propagation."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int = 2,
        dim: int = 64,
        cache_count: int = 8,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) <= 0:
            raise ValueError("CauAir dimensions must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cov_dim = cov_dim

        # These linear summaries are station-shared and therefore independent
        # of N, matching the paper's scalable pre-CachLormer preprocessing.
        self.aqi_summary = nn.Linear(seq_len, dim)
        self.past_cov_summary = nn.Linear(seq_len * cov_dim, dim)
        self.future_cov_summary = nn.Linear(pred_len * cov_dim, dim)
        self.past_association = nn.Linear(dim * 2, dim)
        self.future_association = nn.Linear(dim * 2, dim)
        self.past_cachlormer = CachLormer(dim, heads, cache_count)
        self.future_cachlormer = CachLormer(dim, heads, cache_count)
        self.decoder = nn.Linear(dim, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("CauAir expects (batch, configured seq_len, enc_in)")
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], self.seq_len, 6))
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] != 1 + self.cov_dim:
            raise ValueError(f"CauAir expects exactly {self.cov_dim} historical covariates")
        future_marks = x_mark_enc if x_mark_dec is None else x_mark_dec
        future_marks = coerce_time_length(future_marks, self.pred_len)
        future = future_time_features(future_marks, self.enc_in)
        if future.shape[-1] != self.cov_dim:
            raise ValueError(f"CauAir expects exactly {self.cov_dim} future covariates")

        aqi = self.aqi_summary(x_enc.transpose(1, 2))
        past_cov = history[..., 1:].permute(0, 2, 1, 3).reshape(
            x_enc.shape[0], self.enc_in, self.seq_len * self.cov_dim
        )
        past_cov = self.past_cov_summary(past_cov)
        learned_past = self.past_cachlormer(
            self.past_association(torch.cat([aqi, past_cov], dim=-1))
        )
        future_cov = future.permute(0, 2, 1, 3).reshape(
            x_enc.shape[0], self.enc_in, self.pred_len * self.cov_dim
        )
        future_cov = self.future_cov_summary(future_cov)
        propagated = self.future_cachlormer(
            self.future_association(torch.cat([learned_past, future_cov], dim=-1))
        )
        return self.decoder(propagated).transpose(1, 2)

"""AMD: Adaptive Multi-Scale Decomposition forecaster (MDM, DDI, AMS).

Pipeline on ``x`` of shape ``[batch, channels, seq_len]`` (after RevIN):

* MDM mixes avg-pooled scales ``c^k, ..., c`` top-down: the MLP of each coarse
  scale is added to the next finer scale, giving a "time embedding" of ``x``.
* DDI blocks rewrite ``x`` patch by patch: each ``patch``-wide slice is the
  GELU-linear aggregate of the previous *output* patch plus the input patch,
  optionally followed by an ``alpha``-scaled channel MLP.
* AMS routes every channel's series through ``num_experts`` MLP predictors
  (``seq_len -> pred_len``) with weights from a noisy top-k gate that reads the
  MDM time embedding; a coefficient-of-variation importance loss balances the
  experts and is exposed as ``aux_loss`` (added to the criterion in training).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class FlatBatchNorm(nn.Module):
    """BatchNorm1d over the flattened ``[channels * length]`` features."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.flatten(1)).reshape(x.shape)


class MultiScaleDecomposableMixing(nn.Module):
    """MDM: residual top-down mixing of avg-pooled scales of ``[B, C, L]``."""

    def __init__(self, seq_len: int, channels: int, k: int, c: int, batchnorm: bool) -> None:
        super().__init__()
        self.k = k
        self.norm = FlatBatchNorm(seq_len * channels) if batchnorm else None
        self.windows = [c**i for i in range(k, 0, -1)]
        self.pools = nn.ModuleList(nn.AvgPool1d(w, w) for w in self.windows)
        self.mixers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(seq_len // w, seq_len // w),
                nn.GELU(),
                nn.Linear(seq_len // w, seq_len * c // w),
            )
            for w in self.windows
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        if self.k == 0:
            return x
        scales = [pool(x) for pool in self.pools] + [x]
        for i, mixer in enumerate(self.mixers):
            scales[i + 1] = scales[i + 1] + mixer(scales[i])
        return scales[-1]


class DualDependencyInteraction(nn.Module):
    """DDI: sequential patch aggregation plus an optional channel MLP."""

    def __init__(self, seq_len: int, channels: int, patch: int, alpha: float, dropout: float, batchnorm: bool) -> None:
        super().__init__()
        self.seq_len, self.patch, self.alpha = seq_len, patch, alpha
        self.norm = FlatBatchNorm(seq_len * channels) if batchnorm else None
        self.patch_norm = FlatBatchNorm(patch * channels)
        self.aggregate = nn.Linear(patch, patch)
        self.dropout = nn.Dropout(dropout)
        if alpha > 0.0:
            hidden = 2 ** math.ceil(math.log2(channels))
            self.channel_norm = FlatBatchNorm(patch * channels)
            self.channel_mlp = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        p = self.patch
        pieces = [x[:, :, :p]]
        for start in range(p, self.seq_len, p):
            previous = self.patch_norm(pieces[-1])
            mixed = self.dropout(F.gelu(self.aggregate(previous))) + x[:, :, start : start + p]
            update = mixed
            if self.alpha > 0.0:
                gated = self.channel_norm(mixed).transpose(1, 2)
                update = mixed + self.alpha * self.channel_mlp(gated).transpose(1, 2)
            pieces.append(update)
        return torch.cat(pieces, dim=-1)


class NoisyTopKGate(nn.Module):
    """Noisy linear gate whose logits are sharpened outside the top-k set."""

    def __init__(self, seq_len: int, experts: int, top_k: int, sharpness: float = 10.0) -> None:
        super().__init__()
        self.experts, self.top_k, self.sharpness = experts, top_k, sharpness
        self.gate = nn.Linear(seq_len, experts)
        self.noise = nn.Parameter(torch.zeros(experts, experts))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features``: ``[..., seq_len]``; returns weights ``[..., experts]``."""
        logits = self.gate(features)
        if self.training:
            std = F.softplus(logits @ self.noise) + 1e-5
            logits = logits + torch.randn_like(logits) * std
        kth = torch.kthvalue(logits, self.experts - self.top_k + 1, dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)
        # Outside the top-k set weights are compressed with log(1+p), inside they
        # are expanded with exp(p)-1 before the final softmax.
        sharpened = self.sharpness * torch.where(logits < kth, torch.log1p(probs), torch.expm1(probs))
        return torch.softmax(sharpened, dim=-1)


class AdaptiveMultiPredictorSynthesis(nn.Module):
    """AMS: shared expert MLPs mixed per channel by a time-embedding gate."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        experts: int,
        top_k: int,
        ff_dim: int,
        dropout: float,
        loss_coef: float,
    ) -> None:
        super().__init__()
        self.pred_len, self.loss_coef = pred_len, loss_coef
        self.gating = NoisyTopKGate(seq_len, experts, top_k)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(seq_len, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, pred_len))
            for _ in range(experts)
        )

    def importance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """Sum over channels of ``var(I) / (mean(I)^2 + eps)`` of expert importance.

        ``I`` is the batch sum of the gates, replicated over the horizon axis;
        the variance therefore uses the ``experts * pred_len`` element count.
        """
        experts, horizon = gates.shape[-1], self.pred_len
        if experts == 1:
            return gates.new_zeros(())
        importance = gates.float().sum(0)  # [C, E]
        mean = importance.mean(-1)
        squares = ((importance - mean[:, None]) ** 2).sum(-1)
        variance = squares * horizon / (experts * horizon - 1)
        return self.loss_coef * (variance / (mean**2 + 1e-10)).sum()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor):
        """``x``, ``time_embedding``: ``[B, C, L]``; returns ``([B, C, H], loss)``."""
        gates = self.gating(time_embedding)  # [B, C, E]
        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # [B, C, E, H]
        return (gates.unsqueeze(-1) * outputs).sum(2), self.importance_loss(gates)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_block: int = 1,
        alpha: float = 0.0,
        mix_layer_num: int = 3,
        mix_layer_scale: int = 2,
        patch: int = 16,
        dropout: float = 0.1,
        num_experts: int = 8,
        top_k: int = 2,
        ff_dim: int = 2048,
        moe_loss_coef: float = 1.0,
        use_revin: bool = True,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if seq_len % patch or seq_len % (mix_layer_scale**mix_layer_num):
            raise ValueError("seq_len must be divisible by patch and by mix_layer_scale**mix_layer_num")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.aux_loss: torch.Tensor | None = None
        self.revin = RevIN(enc_in, affine=True, enabled=use_revin)
        self.past_mixing = MultiScaleDecomposableMixing(seq_len, enc_in, mix_layer_num, mix_layer_scale, batchnorm)
        self.blocks = nn.ModuleList(
            DualDependencyInteraction(seq_len, enc_in, patch, alpha, dropout, batchnorm) for _ in range(n_block)
        )
        self.synthesis = AdaptiveMultiPredictorSynthesis(
            seq_len, pred_len, num_experts, top_k, ff_dim, dropout, moe_loss_coef
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        x = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        time_embedding = self.past_mixing(x)
        for block in self.blocks:
            x = block(x)
        y, balance = self.synthesis(x, time_embedding)
        self.aux_loss = balance if self.training else None
        return self.revin(y.transpose(1, 2), "denorm")

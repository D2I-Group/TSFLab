"""Independent inference architecture based on CoRA's published equations.

The implementation retains dynamic low-rank correlations, heterogeneous
positive/negative projections, fusion, and gated residual prediction. It uses
a local linear forecaster instead of a pre-trained TSFM and omits the paper's
training-only H-PCorr contrastive objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN


class _HeterogeneousProjection(nn.Module):
    def __init__(self, d_model: int, enc_in: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.value = nn.Linear(d_model, d_model)
        # A shared scalar bias would cancel exactly under the channel softmax.
        self.channel_score = nn.Linear(d_model, 1, bias=False)
        self.enc_in = enc_in

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(representation)
        weights = self.channel_score(normalized).squeeze(-1).softmax(-1)
        return representation + self.value(normalized) * (weights * self.enc_in).unsqueeze(-1)


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 rank: int = 4, polynomial_order: int = 2,
                 use_revin: bool = True) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("invalid CoRA dimensions")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.rank, self.polynomial_order, self.use_revin = min(rank, enc_in), polynomial_order, use_revin
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.base = nn.Linear(seq_len, pred_len)
        self.encoder = nn.Linear(seq_len, d_model)
        self.coefficient = nn.Linear(d_model, polynomial_order + 1)
        self.polynomial_basis = nn.Parameter(torch.empty(enc_in, self.rank))
        self.invariant_left = nn.Parameter(torch.empty(self.rank, self.rank))
        self.invariant_right = nn.Parameter(torch.empty(self.rank, self.rank))
        self.positive = _HeterogeneousProjection(d_model, enc_in)
        self.negative = _HeterogeneousProjection(d_model, enc_in)
        self.fusion = nn.Linear(2 * d_model, pred_len)
        self.gate = nn.Linear(2 * d_model, pred_len)
        nn.init.normal_(self.polynomial_basis, std=0.1)
        nn.init.normal_(self.invariant_left, std=0.1)
        nn.init.normal_(self.invariant_right, std=0.1)

    def dynamic_correlation(self, x: torch.Tensor, representation: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        covariance = torch.einsum("blc,bld->bcd", centered, centered)
        scale = centered.square().sum(1).sqrt().clamp_min(1e-6)
        pearson = covariance / (scale.unsqueeze(2) * scale.unsqueeze(1))
        coefficients = self.coefficient(representation)
        powers = torch.stack([self.polynomial_basis.pow(order) for order in range(self.polynomial_order + 1)], dim=-1)
        varying = torch.einsum("bco,cro->bcr", coefficients, powers)
        invariant = torch.sigmoid(F.relu(self.invariant_left @ self.invariant_right.transpose(0, 1)))
        return pearson + varying @ invariant @ varying.transpose(1, 2)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        normalized = self.revin(x_enc, "norm") if self.use_revin else x_enc
        channel_history = normalized.transpose(1, 2)
        base = self.base(channel_history)
        representation = self.encoder(channel_history)
        correlation = self.dynamic_correlation(normalized, representation)
        pos = self.positive(representation)
        neg = self.negative(representation)
        pos_context = correlation.clamp_min(0).softmax(-1) @ pos
        neg_context = (-correlation.clamp_max(0)).softmax(-1) @ neg
        fused = torch.cat((pos_context, neg_context), dim=-1)
        correction = self.fusion(fused)
        forecast = torch.sigmoid(self.gate(fused)) * correction + base
        forecast = forecast.transpose(1, 2)
        return self.revin(forecast, "denorm") if self.use_revin else forecast

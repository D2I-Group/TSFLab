"""AROpt: rollout-aware training of an inverted-Transformer short-horizon forecaster.

Independent implementation from Section 3 (Algorithm 1, Eqs. 3-5) of Li, Cheng and
Gu, "AROpt: An Optimization Method for Autoregressive Time Series Forecasting"
(arXiv 2602.02288), after reading the pinned official code
(``LizhengMathAi/AROpt`` at ``c18d9672``, MIT) to resolve omissions; nothing is
copied or imported. The forecaster ``f`` is an iTransformer (variate tokens) built
from the catalog ``embed``, ``self_attention_family`` and ``transformer_encdec``
components; it predicts one patch of ``T = pred_len / rollout_steps`` steps and the
horizon is filled by ``rollout_steps`` autoregressive calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer


class InvertedForecaster(nn.Module):
    """``f(.; theta): R^{S x C} -> R^{T x C}``: per-variate instance normalization,
    whole-series variate tokens, a post-norm Transformer encoder across variates,
    and a linear projector to the ``T``-step patch."""

    def __init__(
        self, seq_len: int, patch_len: int, d_model: int, n_heads: int, e_layers: int,
        d_ff: int, dropout: float, activation: str, use_norm: bool,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, patch_len)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            window, mean, std = standardize(window)
        tokens, _ = self.encoder(self.embedding(window, None))
        patch = self.projector(tokens).transpose(1, 2)  # [B, T, C]
        if self.use_norm:
            patch = destandardize(patch, mean, std)
        return patch


def rollout_objective(errors: list[torch.Tensor], gamma: float, beta: float) -> torch.Tensor:
    """Algorithm 1 lines 4-9 (Eqs. 4-5): ``l = e_1 + sum_k gamma^k ((1 - beta) e_{k+1}
    + beta |e_{k+1} - sg(e_k)|)`` over the per-patch errors ``e_1..e_n``."""
    loss = errors[0]
    for k in range(1, len(errors)):
        penalty = (errors[k] - errors[k - 1].detach()).abs()
        loss = loss + gamma**k * ((1.0 - beta) * errors[k] + beta * penalty)
    return loss


class Model(nn.Module):
    """Short-horizon inverted Transformer rolled out autoregressively over the horizon."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        rollout_steps: int = 4,
        gamma: float = 0.5,
        beta: float = 0.1,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        if pred_len % rollout_steps:
            raise ValueError("pred_len must be divisible by rollout_steps")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.rollout_steps = rollout_steps
        self.patch_len = pred_len // rollout_steps
        self.gamma = gamma
        self.beta = beta
        self.forecaster = InvertedForecaster(
            seq_len, self.patch_len, d_model, n_heads, e_layers, d_ff, dropout, activation, use_norm
        )

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in})")

    def rollout(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Algorithm 1 lines 3 and 7: each call reads the last ``S`` values of the
        history extended by the earlier predicted patches. Fed-back predictions are
        detached (official clarification), so every patch's gradient flows only
        through its own call."""
        patches: list[torch.Tensor] = []
        window = x
        for _ in range(self.rollout_steps):
            patch = self.forecaster(window)
            patches.append(patch)
            window = torch.cat([window, patch.detach()], dim=1)[:, -self.seq_len:]
        return patches

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        self._validate(x_enc)
        return torch.cat(self.rollout(x_enc), dim=1)  # Algorithm 1 line 11

    def patch_errors(self, forecast: torch.Tensor, target: torch.Tensor, criterion) -> list[torch.Tensor]:
        """Per-patch losses ``e_k = L(x_hat_patch_k, x_patch_k)`` over aligned
        ``[B, H, C']`` forecast and target."""
        return [
            criterion(f, t)
            for f, t in zip(forecast.split(self.patch_len, dim=1), target.split(self.patch_len, dim=1))
        ]

    def rollout_loss(self, forecast: torch.Tensor, target: torch.Tensor, criterion) -> torch.Tensor:
        return rollout_objective(self.patch_errors(forecast, target, criterion), self.gamma, self.beta)

"""SOR-Mamba: sequential order-robust Mamba over channel tokens (Lee et al., 2024).

Each channel's look-back window is embedded as one token (Algorithm 1, line 1).
Every layer runs one shared, convolution-free CD-Mamba block over the channel
sequence and over the reversed channel sequence (lines 3-4), adds both to the
residual (line 5) and applies a layer-normalized MLP across the token width
(line 6). A linear head maps each token to the horizon (line 8). Training adds
``lambda * sum_layers d(z1, z2)`` (Eqs. 3-4), the distance between the two
scans, so the unidirectional block becomes robust to the channel order.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.mamba import MambaBlock


class CDMambaBlock(nn.Module):
    """Mamba block without the 1D convolution before the selective SSM (Sec. 4.1, App. D)."""

    def __init__(self, d_model: int, d_state: int, expand: int) -> None:
        super().__init__()
        self.mixer = MambaBlock(
            d_model,
            d_inner=expand * d_model,
            dt_rank=math.ceil(d_model / 16),
            d_conv=1,
            d_state=d_state,
            use_conv=False,
            reference_dt_init=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mixer(tokens)


class SORMambaLayer(nn.Module):
    """Algorithm 1, lines 3-6: shared scans in both channel orders, residual sum, LN-MLP-LN."""

    def __init__(self, d_model: int, d_state: int, d_ff: int, expand: int, dropout: float,
                 mlp_residual: bool) -> None:
        super().__init__()
        self.block = CDMambaBlock(d_model, d_state, expand)
        self.norm_in = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp_residual = mlp_residual

    def scans(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``z1`` in the given channel order and ``z2`` from the reversed order, re-aligned."""
        z1 = self.block(tokens)
        z2 = self.block(tokens.flip(1)).flip(1)
        return z1, z2

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = self.scans(tokens)
        regularizer = F.mse_loss(z1, z2)  # Eq. (3) with d = MSE
        tokens = self.dropout(z1 + z2) + tokens  # line 5
        hidden = self.norm_in(tokens)
        mixed = self.dropout(self.mlp(hidden))
        tokens = self.norm_out(hidden + mixed if self.mlp_residual else mixed)  # line 6
        return tokens, regularizer


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_state: int = 16,
        d_ff: int = 128,
        e_layers: int = 2,
        expand: int = 1,
        dropout: float = 0.1,
        reg_lambda: float = 0.01,
        mlp_residual: bool = False,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("SORMamba lengths, widths and counts must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.reg_lambda = reg_lambda
        self.use_norm = use_norm
        self.embedding = nn.Linear(seq_len, d_model)  # line 1: one token per channel
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            SORMambaLayer(d_model, d_state, d_ff, expand, dropout, mlp_residual) for _ in range(e_layers)
        )
        self.head = nn.Linear(d_model, pred_len)  # line 8

    def forecast_with_regularizer(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the forecast ``[B, H, C]`` and ``sum_i L_reg(z^(i))`` of Eq. (4)."""
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]")
        values = x_enc
        if self.use_norm:
            mean = x_enc.mean(1, keepdim=True).detach()
            scale = x_enc.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
            values = (x_enc - mean) / scale
        tokens = self.embedding_dropout(self.embedding(values.transpose(1, 2)))
        regularizer = tokens.new_zeros(())
        for layer in self.layers:
            tokens, term = layer(tokens)
            regularizer = regularizer + term
        forecast = self.head(tokens).transpose(1, 2)
        if self.use_norm:
            forecast = forecast * scale + mean
        return forecast, regularizer

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast_with_regularizer(x_enc)[0]

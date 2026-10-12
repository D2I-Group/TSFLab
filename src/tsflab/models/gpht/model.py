"""GPHT: generative pretrained hierarchical transformer (KDD 2024), architecture only.

The series is cut into tokens of ``token_len`` steps and modelled auto-regressively
by a stack of decoder-only (causally masked) patch transformers.  Stage ``i``
max-pools its input with kernel ``k_i`` (paper Sec. 3.3), embeds pooled tokens, and a
linear forecast head maps every hidden state to the *next* token (eq. 3).  Stage
``i + 1`` receives the residual ``x - PadFirstToken(out_i)`` (eq. 4) and the stage
outputs are summed (eq. 5) before de-normalization (eq. 6).  Forecasts longer than
one token are produced by rolling the window forward one token at a time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.embed import PatchEmbedding
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer


class HierarchicalStage(nn.Module):
    """Max-pool, causal patch transformer, per-token next-token head (one GPHT stage)."""

    def __init__(
        self,
        token_len: int,
        pooling: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.down_sample = nn.MaxPool1d(pooling)
        patch = token_len // pooling
        self.patch_embedding = PatchEmbedding(d_model, patch, patch, 0, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=dropout), d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.forecast_head = nn.Linear(d_model, token_len)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """``[batch, length, channels]`` -> next-token predictions, same shape."""
        pooled = self.down_sample(values.transpose(1, 2))  # [B, C, length / k]
        tokens, n_vars = self.patch_embedding(pooled)  # [B*C, tokens, d_model]
        hidden, _ = self.encoder(tokens)
        hidden = hidden.reshape(-1, n_vars, hidden.shape[-2], hidden.shape[-1])
        out = self.forecast_head(hidden).reshape(hidden.shape[0], n_vars, -1)
        return out.transpose(1, 2)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        token_len: int = 48,
        pooling_rates: tuple[int, ...] = (8, 4, 2, 1),
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("GPHT dimensions must be positive")
        if seq_len % token_len:
            raise ValueError("seq_len must be a multiple of token_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.token_len = token_len
        self.revin = RevIN(enc_in, affine=False)
        self.stages = nn.ModuleList(
            HierarchicalStage(
                token_len, rate, d_model, d_ff, n_heads, e_layers, dropout, activation
            )
            for rate in pooling_rates
        )

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )

    def next_token_series(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher-forced pass: position ``t`` holds the prediction of ``x[t + token_len]``.

        Returns ``[batch, seq_len, channels]`` in the original scale; its last
        ``token_len`` steps are the one-token forecast (eqs. 3-6).
        """
        self._validate(x)
        residual = self.revin(x, "norm")
        total = torch.zeros_like(residual)
        for stage in self.stages:
            out = stage(residual)
            total = total + out
            # PadFirstToken: right-shift by one token so out predicts the same token.
            shifted = torch.cat(
                (torch.zeros_like(out[:, : self.token_len]), out[:, : -self.token_len]), dim=1
            )
            residual = residual - shifted
        return self.revin(total, "denorm")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        window = x_enc
        pieces = []
        steps = -(-self.pred_len // self.token_len)
        for _ in range(steps):
            token = self.next_token_series(window)[:, -self.token_len :]
            pieces.append(token)
            window = torch.cat((window, token), dim=1)[:, -self.seq_len :]
        return torch.cat(pieces, dim=1)[:, : self.pred_len]

    def training_objective(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Next-token MSE on every position: targets are ``x[token_len:]`` plus the next token."""
        if target.shape[0] != x.shape[0] or target.shape[2] != x.shape[2]:
            raise ValueError("GPHT trains on all channels and needs a matching target")
        if target.shape[1] < self.token_len:
            raise ValueError("training needs pred_len >= token_len for the next-token target")
        series = self.next_token_series(x)
        shifted = torch.cat((x[:, self.token_len :], target[:, : self.token_len]), dim=1)
        return torch.mean((series - shifted) ** 2)

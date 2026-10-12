"""Clean-room PMDformer from the public ICLR 2026 equations.

The code implements patch-mean decoupling (1--3), proximal variable attention
(4--5), and trend-restoration attention (6--9). It was designed from the paper
without consulting the linked reference implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class TrendRestorationAttention(nn.Module):
    """Shape-only Q/K attention with patch means restored into V."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, shape_tokens: torch.Tensor, patch_means: torch.Tensor) -> torch.Tensor:
        q, k = self.query(shape_tokens), self.key(shape_tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        weights = scores.softmax(dim=-1)
        values = self.value(shape_tokens) + patch_means.unsqueeze(-1)
        attended = self.output(torch.matmul(weights, values))
        hidden = self.norm1(shape_tokens + self.dropout(attended))
        return self.norm2(hidden + self.dropout(self.ffn(hidden)))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        patch_len: int = 16,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len = min(patch_len, seq_len)
        self.num_patches = math.ceil(seq_len / self.patch_len)
        self.pad_left = self.num_patches * self.patch_len - seq_len
        self.revin = RevIN(enc_in, enabled=use_revin)
        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, 1, self.num_patches, d_model))
        self.proximal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.proximal_norm1 = nn.LayerNorm(d_model)
        self.proximal_ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.proximal_norm2 = nn.LayerNorm(d_model)
        self.temporal_attention = TrendRestorationAttention(d_model, dropout)
        self.projection = nn.Linear(self.num_patches * d_model, pred_len)

    def patch_mean_decouple(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history = x.transpose(1, 2)
        if self.pad_left:
            history = F.pad(history, (self.pad_left, 0), mode="replicate")
        patches = history.unfold(-1, self.patch_len, self.patch_len)
        means = patches.mean(dim=-1)
        residuals = patches - means.unsqueeze(-1)
        return residuals, means

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        residuals, means = self.patch_mean_decouple(normalized)
        tokens = self.patch_projection(residuals) + self.position

        proximal = tokens[:, :, -1]
        mixed, _ = self.proximal_attention(proximal, proximal, proximal, need_weights=False)
        proximal = self.proximal_norm1(proximal + mixed)
        proximal = self.proximal_norm2(proximal + self.proximal_ffn(proximal))
        tokens = torch.cat((tokens[:, :, :-1], proximal.unsqueeze(2)), dim=2)

        temporal = self.temporal_attention(tokens, means)
        restored = temporal + means.unsqueeze(-1)
        forecast = self.projection(restored.flatten(-2, -1)).transpose(1, 2)
        return self.revin(forecast, "denorm")

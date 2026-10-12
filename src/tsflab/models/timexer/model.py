"""Clean-room TimeXer with endogenous patches and exogenous cross-attention."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN


class EndogenousEmbedding(nn.Module):
    def __init__(self, patch_len, patch_count, width, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value = nn.Linear(patch_len, width)
        self.position = nn.Parameter(torch.randn(patch_count, width) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,E,L), output: (B*E,P+1,D)
        batch, endogenous, length = x.shape
        patches = math.ceil(length / self.patch_len)
        padded = F.pad(x, (0, patches*self.patch_len-length))
        tokens = self.value(padded.unfold(-1, self.patch_len, self.patch_len)) + self.position[:patches]
        tokens = tokens.reshape(batch*endogenous, patches, -1)
        return self.dropout(torch.cat((tokens, self.global_token.expand(batch*endogenous, -1, -1)), 1))


class ExogenousEmbedding(nn.Module):
    """Represent each external variable and the calendar as cross-attention tokens."""
    def __init__(self, seq_len, width):
        super().__init__()
        self.value = nn.Linear(seq_len, width)
        self.calendar = nn.Linear(4, width)

    def forward(self, all_series, marks, endogenous_count):
        batch, variables, _ = all_series.shape
        tokens = self.value(all_series)
        if marks is None:
            position = torch.linspace(0, 1, all_series.shape[-1], device=all_series.device, dtype=all_series.dtype)
            calendar = torch.stack((position.mean(), position.square().mean(), torch.sin(2*torch.pi*position).mean(), torch.cos(2*torch.pi*position).mean())).expand(batch, -1)
        elif marks.shape[-1] >= 6:
            calendar = torch.stack((marks[...,1].mean(1)/12, marks[...,2].mean(1)/31, marks[...,3].mean(1)/7, marks[...,4].mean(1)/24), -1)
        else:
            calendar = F.pad(marks[..., :4].mean(1), (0, max(0, 4-marks.shape[-1])))[:, :4]
        tokens = torch.cat((tokens, self.calendar(calendar).unsqueeze(1)), 1)
        return tokens[:, None].expand(-1, endogenous_count, -1, -1).reshape(batch*endogenous_count, variables+1, -1)


class TimeXerLayer(nn.Module):
    def __init__(self, width, heads, hidden, dropout, activation):
        super().__init__()
        self.patch_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.exogenous_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.feedforward = nn.Sequential(nn.Linear(width, hidden), act, nn.Dropout(dropout), nn.Linear(hidden, width))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.last_cross_attention = None

    def forward(self, endogenous, exogenous):
        attended, _ = self.patch_attention(endogenous, endogenous, endogenous, need_weights=False)
        endogenous = self.norm1(endogenous + attended)
        global_token = endogenous[:, -1:]
        external, weights = self.exogenous_attention(global_token, exogenous, exogenous, need_weights=True)
        self.last_cross_attention = weights
        global_token = self.norm2(global_token + external)
        endogenous = torch.cat((endogenous[:, :-1], global_token), 1)
        return self.norm3(endogenous + self.feedforward(endogenous))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M", d_model=128,
                 n_heads=8, e_layers=2, d_ff=256, patch_len=16, dropout=0.1,
                 activation="gelu", use_norm=True):
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("invalid TimeXer dimensions")
        self.seq_len, self.pred_len, self.enc_in, self.features = seq_len, pred_len, enc_in, features
        self.endogenous_count = enc_in if features == "M" else 1
        self.patch_count = math.ceil(seq_len / patch_len)
        self.revin = RevIN(enc_in, enabled=use_norm)
        self.endogenous_embedding = EndogenousEmbedding(patch_len, self.patch_count, d_model, dropout)
        self.exogenous_embedding = ExogenousEmbedding(seq_len, d_model)
        self.layers = nn.ModuleList(TimeXerLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers))
        self.head = nn.Sequential(nn.Flatten(-2), nn.Linear((self.patch_count+1)*d_model, pred_len))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        normalized = self.revin(x_enc, "norm")
        all_series = normalized.transpose(1, 2)
        endogenous = all_series if self.features == "M" else all_series[:, -1:]
        tokens = self.endogenous_embedding(endogenous)
        external = self.exogenous_embedding(all_series, x_mark_enc, self.endogenous_count)
        for layer in self.layers:
            tokens = layer(tokens, external)
        forecast = self.head(tokens).reshape(x_enc.shape[0], self.endogenous_count, self.pred_len).transpose(1, 2)
        if self.features == "M":
            return self.revin(forecast, "denorm")
        # MS/S output is the designated endogenous (last) variable.
        full = forecast.new_zeros(x_enc.shape[0], self.pred_len, self.enc_in)
        full[..., -1:] = forecast
        return self.revin(full, "denorm")[..., -1:]

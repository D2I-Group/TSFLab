"""Clean-room Crossformer based on DSW embedding, TSA, and hierarchy."""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._inputs import check_window


def dsw_embed(x, seg_len, projection):
    """Dimension-segment-wise embedding: ``[B,L,C] -> [B,C,S,D]``."""
    padding = (-x.shape[1]) % seg_len
    if padding:
        x = torch.cat((x[:, :1].expand(-1, padding, -1), x), 1)
    return projection(x.transpose(1, 2).unfold(-1, seg_len, seg_len))


class TwoStageAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, routers, dropout):
        super().__init__()
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.sender = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.receiver = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.routers = nn.Parameter(torch.randn(1, routers, d_model) * 0.02)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        batch, channels, segments, width = x.shape
        temporal = x.reshape(batch * channels, segments, width)
        update, _ = self.time_attention(temporal, temporal, temporal, need_weights=False)
        x = self.norm1(x + update.reshape_as(x))
        variables = x.transpose(1, 2).reshape(batch * segments, channels, width)
        routers = self.routers.expand(batch * segments, -1, -1)
        messages, _ = self.sender(routers, variables, variables, need_weights=False)
        update, _ = self.receiver(variables, messages, messages, need_weights=False)
        x = self.norm2(x + update.reshape(batch, segments, channels, width).transpose(1, 2))
        return self.norm3(x + self.ffn(x))


class SegmentMerge(nn.Module):
    def __init__(self, width, window):
        super().__init__()
        self.window = window
        self.projection = nn.Linear(window * width, width)

    def forward(self, x):
        padding = (-x.shape[2]) % self.window
        if padding:
            x = torch.cat((x, x[:, :, -1:].expand(-1, -1, padding, -1)), 2)
        batch, channels, segments, width = x.shape
        return self.projection(x.reshape(batch, channels, segments // self.window, self.window * width))


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        seg_len=12,
        win_size=2,
        factor=10,
        dropout=0.1,
    ):
        super().__init__()
        if seg_len > seq_len:
            raise ValueError("invalid segment hierarchy")
        self.seq_len, self.pred_len, self.enc_in, self.seg_len = seq_len, pred_len, enc_in, seg_len
        count = math.ceil(seq_len / seg_len)
        self.segment_projection = nn.Linear(seg_len, d_model)
        self.position = nn.Parameter(torch.randn(1, enc_in, count, d_model) * 0.02)
        self.layers, self.mergers, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        current = count
        for level in range(e_layers):
            if level:
                self.mergers.append(SegmentMerge(d_model, win_size))
                current = math.ceil(current / win_size)
            self.layers.append(TwoStageAttention(d_model, n_heads, d_ff, factor, dropout))
            self.heads.append(nn.Linear(current * d_model, pred_len))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        x = dsw_embed(x_enc, self.seg_len, self.segment_projection) + self.position
        forecasts = []
        for level, layer in enumerate(self.layers):
            if level:
                x = self.mergers[level - 1](x)
            x = layer(x)
            forecasts.append(self.heads[level](x.flatten(2)).transpose(1, 2))
        return torch.stack(forecasts).mean(0)

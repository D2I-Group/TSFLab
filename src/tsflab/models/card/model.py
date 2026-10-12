"""Independent CARD forecast implementation from the published method.

This follows the paper's channel-aligned attention and token-blend ideas. It
was written from the paper description; unlicensed reference code was not used.
"""
from __future__ import annotations
import torch
from torch import nn


def exponential_smooth(tokens: torch.Tensor, alpha: float) -> torch.Tensor:
    """Causal EMA over the token axis of a ``[B,C,P,D]`` tensor."""
    states = [tokens[:, :, 0]]
    for index in range(1, tokens.shape[2]):
        states.append(alpha * tokens[:, :, index] + (1.0 - alpha) * states[-1])
    return torch.stack(states, dim=2)


class ChannelAlignedBlock(nn.Module):
    """Blend temporal-token attention with aligned cross-channel attention."""
    def __init__(self, d_model, n_heads, d_ff, dropout, alpha):
        super().__init__()
        self.alpha = alpha
        self.token_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.blend = nn.Linear(2 * d_model, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, tokens):
        batch, channels, patches, width = tokens.shape
        smooth = exponential_smooth(tokens, self.alpha)
        flat = smooth.reshape(batch * channels, patches, width)
        temporal, _ = self.token_attention(flat, flat, tokens.reshape_as(flat), need_weights=False)
        temporal = temporal.reshape_as(tokens)
        aligned = tokens.transpose(1, 2).reshape(batch * patches, channels, width)
        cross, _ = self.channel_attention(aligned, aligned, aligned, need_weights=False)
        cross = cross.reshape(batch, patches, channels, width).transpose(1, 2)
        gate = torch.sigmoid(self.blend(torch.cat((temporal, cross), -1)))
        tokens = self.norm1(tokens + gate * temporal + (1.0 - gate) * cross)
        return self.norm2(tokens + self.ffn(tokens))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M", patch_len=16, stride=8,
                 d_model=128, n_heads=8, e_layers=2, d_ff=256, dropout=0.1,
                 alpha=0.5, use_statistic=False):
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must be within the history")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        count = 1 + (seq_len - patch_len) // stride
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.randn(1, 1, count, d_model) * 0.02)
        self.stat_projection = nn.Linear(2, d_model) if use_statistic else None
        self.blocks = nn.ModuleList([ChannelAlignedBlock(d_model, n_heads, d_ff, dropout, alpha) for _ in range(e_layers)])
        self.head = nn.Linear((count + int(use_statistic)) * d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        series = x_enc.transpose(1, 2)
        tokens = self.patch_projection(series.unfold(-1, self.patch_len, self.stride)) + self.position
        if self.stat_projection is not None:
            stats = torch.stack((series.mean(-1), series.std(-1, unbiased=False)), -1)
            tokens = torch.cat((self.stat_projection(stats).unsqueeze(2), tokens), 2)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens.flatten(2)).transpose(1, 2)

"""Independent full-FFT Fredformer forecasting path; see reference.md."""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class ChannelBlock(nn.Module):
    """Pre-normalized channel attention and residual feed-forward block."""

    def __init__(self, width, heads, head_dim, feedforward, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * heads * head_dim)
        self.attention_out = nn.Sequential(nn.Linear(heads * head_dim, width), nn.Dropout(dropout))
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        batch, channels, _ = tokens.shape
        q, k, v = self.qkv(self.norm_attention(tokens)).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, channels, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        weights = (q @ k.transpose(-2, -1) / self.head_dim**0.5).softmax(-1)
        mixed = (weights @ v).transpose(1, 2).reshape(batch, channels, -1)
        tokens = tokens + self.attention_out(mixed)
        return tokens + self.feedforward(self.norm_feedforward(tokens))


class ForecastHead(nn.Module):
    def __init__(self, width, horizon):
        super().__init__()
        self.residual_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.project = nn.Linear(width, horizon)

    def forward(self, patches):
        values = patches.flatten(-2)
        for layer in self.residual_layers:
            values = values + layer(values).relu()
        return self.project(values)


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        band_width=16,
        model_width=48,
        depth=2,
        heads=6,
        feedforward=128,
        dropout=0.2,
        revin=True,
        affine=True,
        subtract_last=False,
        head_dropout=0.0,
        stride=16,
        head_dim=64,
    ):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.band_width, self.stride = band_width, stride
        self.revin = RevIN(enc_in, affine=affine, subtract_last=subtract_last, enabled=revin)
        self.input_projection = nn.Sequential(nn.Linear(2 * band_width, model_width), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(
            [ChannelBlock(model_width, heads, head_dim, feedforward, dropout) for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(nn.Dropout(dropout), nn.Linear(model_width, 2 * model_width))
        self.real_projection = nn.Linear(2 * model_width, 2 * model_width)
        self.imag_projection = nn.Linear(2 * model_width, 2 * model_width)
        patch_count = (seq_len - band_width) // stride + 1
        self.real_head = ForecastHead(2 * model_width * patch_count, pred_len)
        self.imag_head = ForecastHead(2 * model_width * patch_count, pred_len)
        self.inverse_projection = nn.Linear(2 * pred_len, pred_len)
        # Shared official head does not apply its declared head_dropout.

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        spectrum = torch.fft.fft(normalized.transpose(1, 2), dim=-1)
        patches = torch.cat(
            (
                spectrum.real.unfold(-1, self.band_width, self.stride),
                spectrum.imag.unfold(-1, self.band_width, self.stride),
            ),
            dim=-1,
        )
        batch, channels, count, _ = patches.shape
        tokens = patches.permute(0, 2, 1, 3).reshape(batch * count, channels, -1)
        tokens = self.input_projection(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_projection(tokens).reshape(batch, count, channels, -1).permute(0, 2, 1, 3)
        real = self.real_head(self.real_projection(tokens))
        imag = self.imag_head(self.imag_projection(tokens))
        signal = torch.fft.ifft(torch.complex(real, imag), dim=-1)
        forecast = self.inverse_projection(torch.cat((signal.real, signal.imag), dim=-1)).transpose(1, 2)
        return self.revin(forecast, "denorm")

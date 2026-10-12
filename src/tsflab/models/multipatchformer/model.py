"""Independent MultiPatchFormer with multi-scale patches and SAR decoding."""
from __future__ import annotations
import torch
from torch import nn

from tsflab.models._components.instance_norm import destandardize, standardize


class PatchScale(nn.Module):
    def __init__(self, seq_len, patch_len, stride, width):
        super().__init__()
        self.patch_len, self.stride = min(seq_len, patch_len), min(seq_len, stride)
        self.projection = nn.Linear(self.patch_len, width)

    def forward(self, series):
        patches = series.unfold(-1, self.patch_len, self.stride)
        return self.projection(patches)


class SemiAutoregressiveHead(nn.Module):
    """Predict horizon groups while conditioning each group on earlier groups."""
    def __init__(self, width, horizon, groups=8):
        super().__init__()
        groups = min(groups, horizon)
        sizes = [horizon // groups + int(i < horizon % groups) for i in range(groups)]
        self.layers = nn.ModuleList()
        emitted = 0
        for size in sizes:
            self.layers.append(nn.Linear(width + emitted, size))
            emitted += size

    def forward(self, tokens):
        chunks = []
        for layer in self.layers:
            chunks.append(layer(torch.cat((tokens, *chunks), -1) if chunks else tokens))
        return torch.cat(chunks, -1)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M", label_len=0, d_model=64,
                 n_heads=4, e_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        if d_model % 4 or d_model % n_heads or e_layers < 1:
            raise ValueError("d_model must divide four scales and attention heads")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        scale_width = d_model // 4
        self.scales = nn.ModuleList([PatchScale(seq_len, p, s, scale_width) for p, s in ((8, 8), (16, 8), (24, 7), (32, 6))])
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
            for _ in range(e_layers)
        ])
        self.channel_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.channel_projection = nn.Linear(d_model, d_model)
        self.head = SemiAutoregressiveHead(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        series, mean, std = standardize(x_enc)
        series = series.transpose(1, 2)
        batch, channels = series.shape[:2]
        target_tokens = max(scale(series).shape[2] for scale in self.scales)
        embeddings = []
        for scale in self.scales:
            tokens = scale(series)
            token_count, scale_width = tokens.shape[2:]
            tokens = torch.nn.functional.interpolate(
                tokens.reshape(batch * channels, token_count, scale_width).transpose(1, 2),
                size=target_tokens, mode="linear", align_corners=False,
            ).transpose(1, 2).reshape(batch, channels, target_tokens, scale_width)
            embeddings.append(tokens)
        tokens = torch.cat(embeddings, -1)
        batch, channels, patches, width = tokens.shape
        tokens = tokens.reshape(batch * channels, patches, width)
        for layer in self.temporal_layers:
            tokens = layer(tokens)
        channels_tokens = self.channel_projection(tokens.mean(1).reshape(batch, channels, width))
        channels_tokens = self.channel_layer(channels_tokens)
        forecast = self.head(channels_tokens).transpose(1, 2)
        return destandardize(forecast, mean, std)

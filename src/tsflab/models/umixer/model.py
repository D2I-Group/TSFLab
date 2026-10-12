"""Clean-room U-Mixer with U-shaped patch mixing and stationarity correction."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class AxisMixer(nn.Module):
    """Mix patch positions and embedding features on their correct axes."""
    def __init__(self, patch_count, width, dropout):
        super().__init__()
        # The following LayerNorm removes a bias that is constant across the
        # feature axis. Keep the final patch projection bias-free so the state
        # dict contains no structurally inactive parameter.
        self.patch_mlp = nn.Sequential(
            nn.Linear(patch_count, patch_count),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_count, patch_count, bias=False),
        )
        self.feature_mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*width, width))
        self.patch_norm = nn.LayerNorm(width)
        self.feature_norm = nn.LayerNorm(width)

    def forward(self, x):
        x = self.patch_norm(x + self.patch_mlp(x.transpose(-1, -2)).transpose(-1, -2))
        return self.feature_norm(x + self.feature_mlp(x))


class StationarityCorrection(nn.Module):
    """Restore relative autocorrelation energy removed by deep processing."""
    def __init__(self, width):
        super().__init__()
        self.channel_gate = nn.Sequential(nn.Linear(width, width), nn.Sigmoid())
        self.last_factor = None

    def forward(self, original, processed):
        original_power = torch.fft.rfft(original, dim=-2).abs().square().mean(-2, keepdim=True)
        processed_power = torch.fft.rfft(processed, dim=-2).abs().square().mean(-2, keepdim=True)
        ratio = ((original_power + 1e-5) / (processed_power + 1e-5)).sqrt().mean(-2, keepdim=True)
        factor = 1 + self.channel_gate(original.mean(-2)).unsqueeze(-2) * (ratio - 1)
        self.last_factor = factor
        return processed * factor


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, label_len=0,
                 features="M", d_model=64, e_layers=2, patch_len=16,
                 stride=8, dropout=0.1):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        self.patch_count = max(1, math.ceil(max(0, seq_len-patch_len)/stride)+1)
        counts = [self.patch_count]
        for _ in range(e_layers):
            counts.append(max(1, math.ceil(counts[-1]/2)))
        self.revin = RevIN(enc_in)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.down_mixers = nn.ModuleList(AxisMixer(counts[i], d_model, dropout) for i in range(e_layers))
        self.bottleneck = AxisMixer(counts[-1], d_model, dropout)
        self.up_mixers = nn.ModuleList(AxisMixer(counts[i], d_model, dropout) for i in reversed(range(e_layers)))
        self.skip_fusion = nn.ModuleList(nn.Linear(2*d_model, d_model) for _ in range(e_layers))
        self.correction = StationarityCorrection(d_model)
        self.head = nn.Linear(self.patch_count*d_model, pred_len)

    def _patch(self, x):
        needed = (self.patch_count-1)*self.stride + self.patch_len
        x = F.pad(x, (0, max(0, needed-x.shape[-1])))
        return self.patch_embedding(x.unfold(-1, self.patch_len, self.stride)[..., :self.patch_count, :])

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm").transpose(1, 2)
        original = self._patch(normalized)
        hidden, skips = original, []
        for mixer in self.down_mixers:
            hidden = mixer(hidden)
            skips.append(hidden)
            if hidden.shape[-2] > 1:
                hidden = F.avg_pool1d(hidden.flatten(0, 1).transpose(1, 2), 2, ceil_mode=True).transpose(1, 2).reshape(*hidden.shape[:2], -1, hidden.shape[-1])
        hidden = self.bottleneck(hidden)
        for mixer, fusion, skip in zip(self.up_mixers, self.skip_fusion, reversed(skips)):
            hidden = F.interpolate(hidden.flatten(0, 1).transpose(1, 2), size=skip.shape[-2], mode="linear", align_corners=False).transpose(1, 2).reshape_as(skip)
            hidden = mixer(fusion(torch.cat((hidden, skip), -1)))
        corrected = self.correction(original, hidden)
        forecast = self.head(corrected.flatten(-2)).transpose(1, 2)
        return self.revin(forecast, "denorm")

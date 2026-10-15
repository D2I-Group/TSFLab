"""Independent HDMixer implementation from the AAAI paper architecture.

Length-Extendable Patching learns a center and width adjustment for every
nominal patch. Hierarchical Dependency Explorer blocks then mix within-patch,
across-patch, and across-variable axes with separate MLPs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class LengthExtendablePatcher(nn.Module):
    """Sample learned variable-length patches by differentiable interpolation."""

    def __init__(self, seq_len: int, patch_len: int, stride: int, deform_range: float):
        super().__init__()
        self.seq_len, self.patch_len = seq_len, patch_len
        self.patch_count = max(1, seq_len // stride)
        centers = torch.linspace(0, seq_len - 1, self.patch_count)
        self.register_buffer("centers", centers)
        self.offsets = nn.Linear(seq_len, 2 * self.patch_count)
        self.deform_range = deform_range

    def sampling_grid(self, series: torch.Tensor) -> torch.Tensor:
        shifts = torch.tanh(self.offsets(series)).view(-1, self.patch_count, 2)
        center = self.centers + shifts[..., 0] * self.deform_range * self.seq_len
        width = self.patch_len * torch.exp(shifts[..., 1] * self.deform_range)
        unit = torch.linspace(-0.5, 0.5, self.patch_len, device=series.device)
        locations = center.unsqueeze(-1) + width.unsqueeze(-1) * unit
        return 2 * locations / max(1, self.seq_len - 1) - 1

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        series = values.transpose(1, 2).reshape(batch * channels, length)
        horizontal = self.sampling_grid(series)
        vertical = torch.zeros_like(horizontal)
        grid = torch.stack((horizontal, vertical), -1)
        sampled = F.grid_sample(
            series[:, None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(1)
        return sampled.reshape(batch, channels, self.patch_count, self.patch_len)


class AxisMixer(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width), nn.Dropout(dropout)
        )

    def forward(self, values):
        return values + self.net(self.norm(values))


class HierarchicalDependencyBlock(nn.Module):
    """HDE with distinct local-time, long-time, variable and channel mixers."""

    def __init__(
        self, variables, patches, patch_len, width, hidden, dropout, mix_time=True, mix_variable=True, mix_channel=True
    ):
        super().__init__()
        self.local = AxisMixer(patch_len, max(patch_len, hidden // 4), dropout) if mix_time else nn.Identity()
        self.patch = AxisMixer(patches, max(patches, hidden // 4), dropout) if mix_time else nn.Identity()
        self.variable = AxisMixer(variables, max(variables, hidden // 4), dropout) if mix_variable else nn.Identity()
        self.channel = AxisMixer(width, hidden, dropout) if mix_channel else nn.Identity()

    def forward(self, values):
        # B,C,P,W,D; axes are moved to the last position for each mixer.
        values = self.local(values.transpose(-1, -2)).transpose(-1, -2)
        values = self.patch(values.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        values = self.variable(values.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return self.channel(values)


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=128,
        d_ff=256,
        e_layers=3,
        patch_len=16,
        stride=8,
        dropout=0.1,
        head_dropout=0.0,
        revin=True,
        affine=True,
        subtract_last=False,
        deform_range=0.25,
        mix_time=True,
        mix_variable=True,
        mix_channel=True,
    ):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in, affine=affine, subtract_last=subtract_last, enabled=revin)
        self.patcher = LengthExtendablePatcher(seq_len, patch_len, stride, deform_range)
        self.embedding = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(
            [
                HierarchicalDependencyBlock(
                    enc_in,
                    self.patcher.patch_count,
                    patch_len,
                    d_model,
                    d_ff,
                    dropout,
                    mix_time,
                    mix_variable,
                    mix_channel,
                )
                for _ in range(e_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Flatten(-3),
            nn.Dropout(head_dropout),
            nn.Linear(self.patcher.patch_count * patch_len * d_model, pred_len),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        hidden = self.embedding(self.patcher(normalized).unsqueeze(-1))
        for layer in self.layers:
            hidden = layer(hidden)
        forecast = self.head(hidden).transpose(1, 2)
        return self.revin(forecast, "denorm")

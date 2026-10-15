"""PatchMLP building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average block to highlight trend."""

    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        return self.avg(x)


class SeriesDecomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EmbLayer(nn.Module):
    """Patch embedding block for PatchMLP."""

    def __init__(self, patch_len: int, patch_step: int, seq_len: int, d_model: int) -> None:
        super().__init__()
        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model = d_model // patch_num
        self.patch_len = patch_len
        self.patch_step = patch_step
        self.ff = nn.Linear(patch_len, self.d_model)
        self.flatten = nn.Flatten(start_dim=-2)
        self.ff_1 = nn.Linear(self.d_model * patch_num, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)
        return self.ff_1(x)


class Emb(nn.Module):
    """Multi-scale patch embedding."""

    def __init__(self, seq_len: int, d_model: int, patch_len: list[int]) -> None:
        super().__init__()
        patch_step = patch_len
        d_model = d_model // 4
        self.emb_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.emb_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.emb_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.emb_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_x1 = self.emb_1(x)
        s_x2 = self.emb_2(x)
        s_x3 = self.emb_3(x)
        s_x4 = self.emb_4(x)
        return torch.cat([s_x1, s_x2, s_x3, s_x4], dim=-1)


class Encoder(nn.Module):
    """Intra-variable MLP with optional smooth-branch channel mixing."""

    def __init__(self, d_model: int, enc_in: int, *, channel_mixing: bool) -> None:
        super().__init__()
        self.channel_mixing = channel_mixing
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        if channel_mixing:
            self.norm2 = nn.LayerNorm(d_model)
            self.ff2 = nn.Sequential(
                nn.Linear(enc_in, enc_in),
                nn.GELU(),
                nn.Dropout(0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        if not self.channel_mixing:
            return y_0
        y_1 = y_0.permute(0, 2, 1)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        y_2 = y_1 * y_0 + x
        return self.norm2(y_2)

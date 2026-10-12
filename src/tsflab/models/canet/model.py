"""Clean-room local implementation of CANet's chrono-adaptive multi-scale forecaster.

CANet runs several parallel branches, one per patch size, each converting the
(instance-normalized) lookback window into overlapping patches, filtering them
in the frequency domain (:class:`AdaptiveSpectralBlock`), re-normalizing with
a *non-stationary adaptive* normalization (NSAN) that restores per-sample
mean/standard-deviation "style" instead of a fixed learned affine transform,
and mixing them with a dual-path convolution (:class:`InteractiveConvolutionalBlock`).
All branches' outputs are concatenated and linearly projected to the horizon
(paper Section 3).

This local implementation keeps the paper's default configuration (LayerNorm
before the spectral block, NSAN after it) and its two default patch scales;
the alternative BatchNorm/InstanceNorm/DAIN normalization ablations and the
Stacked-Kronecker-Product rank-reducing projection are not reproduced (a
plain dense projection is used instead) -- see the model card for the full
list of differences.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.adain_style_norm import AdaptiveInstanceNorm1d
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._inputs import check_window


class Patchifier(nn.Module):
    """Split the last axis into overlapping patches with 50% stride."""

    def __init__(self, sequence_length: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = max(1, patch_size // 2)
        self.num_patches = (sequence_length - patch_size) // self.stride + 1
        if self.num_patches < 1:
            raise ValueError("sequence_length must be at least patch_size")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)


class MeanStdProjector(nn.Module):
    """Project scalar per-variate mean/std statistics into embedding space."""

    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.mean_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.Dropout(dropout))
        self.std_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.Dropout(dropout))

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # mean/std: (batch, 1, m) -> (batch * m, 1, 1) -> (batch * m, 1, embed_dim)
        batch, _, m = mean.shape
        mean = mean.transpose(1, 2).reshape(batch * m, 1, 1)
        std = std.transpose(1, 2).reshape(batch * m, 1, 1)
        return self.mean_proj(mean), self.std_proj(std)


class StyleBlendingGate(nn.Module):
    """Convexly blend two (mean, std) style pairs and project each."""

    def __init__(self, embed_dim: int, blend_ratio: float, affine: bool) -> None:
        super().__init__()
        self.blend_ratio = blend_ratio
        self.affine = affine
        if affine:
            self.mean_shift_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.mean_shift_b = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.std_scale_a = nn.Parameter(torch.ones(1, 1, embed_dim))
            self.std_scale_b = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.mean_linear = nn.Linear(embed_dim, embed_dim)
        self.std_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, mean_a, std_a, mean_b, std_b):
        if self.affine:
            mean_a = mean_a * self.mean_shift_a
            mean_b = mean_b * self.mean_shift_b
            std_a = std_a * self.std_scale_a
            std_b = std_b * self.std_scale_b
        mean = self.blend_ratio * mean_a + (1 - self.blend_ratio) * mean_b
        std = self.blend_ratio * std_a + (1 - self.blend_ratio) * std_b
        return self.mean_linear(mean), self.std_linear(std)


class AdaptiveSpectralBlock(nn.Module):
    """Learned complex frequency-domain filter with an adaptive high-pass mask."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def _adaptive_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        median_energy = energy.flatten(1).median(dim=1, keepdim=True)[0]
        normalized_energy = energy / (median_energy + 1e-6)
        mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach()
        mask = mask + self.threshold_param
        return mask.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        x_fft = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        weighted = x_fft * torch.view_as_complex(self.complex_weight)
        masked = x_fft * self._adaptive_mask(x_fft).to(x_fft.dtype)
        weighted = weighted + masked * torch.view_as_complex(self.complex_weight_high)
        return torch.fft.irfft(weighted, n=length, dim=1, norm="ortho").to(x.dtype)


class InteractiveConvolutionalBlock(nn.Module):
    """Dual-path (kernel 1 and 3) convolution fused by cross-multiplication."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.conv3 = nn.Conv1d(dim, dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        path1 = self.conv1(x)
        path2 = self.conv2(x)
        gated1, gated2 = self.drop(self.act(path1)), self.drop(self.act(path2))
        fused = path1 * gated2 + path2 * gated1
        return self.conv3(fused).transpose(1, 2)


class CANetLayer(nn.Module):
    """One patch-size branch: patch, spectral-filter, NSAN-normalize, convolve."""

    def __init__(self, seq_len: int, patch_size: int, embed_dim: int, dropout: float, blend_ratio: float) -> None:
        super().__init__()
        self.patchifier = Patchifier(seq_len, patch_size)
        self.position = positional_encoding("exp2d", True, self.patchifier.num_patches, patch_size)
        self.input_layer1 = nn.Sequential(nn.Linear(patch_size, embed_dim), nn.Dropout(dropout))
        self.input_layer2 = nn.Sequential(nn.Linear(patch_size, embed_dim), nn.Dropout(dropout))
        self.mean_std_proj = MeanStdProjector(embed_dim, dropout)
        self.style_gate = StyleBlendingGate(embed_dim, blend_ratio, affine=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.asb = AdaptiveSpectralBlock(embed_dim)
        self.nsan = AdaptiveInstanceNorm1d()
        self.icb = InteractiveConvolutionalBlock(embed_dim, dropout)

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, m, seq_len)
        patched = self.patchifier(x) + self.position
        style = patched.detach().clone()
        batch, m, num_patches, patch_size = patched.shape
        patched = patched.reshape(batch * m, num_patches, patch_size)
        style = style.reshape(batch * m, num_patches, patch_size)

        patched = self.input_layer1(patched)
        style = self.input_layer2(style)
        style_mean = style.mean(dim=1, keepdim=True)
        style_std = style.std(dim=1, keepdim=True) + 1e-5

        patched = self.norm1(patched)
        patched = self.asb(patched)

        proj_mean, proj_std = self.mean_std_proj(mean, std)
        blend_mean, blend_std = self.style_gate(proj_mean, proj_std, style_mean, style_std)
        patched = self.nsan(patched, blend_mean, blend_std)

        patched = self.icb(patched)
        return style + patched


class Model(nn.Module):
    """Multi-scale patch forecaster with chrono-adaptive normalization."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_sizes: tuple[int, ...] = (8, 64),
        embed_dim: int = 32,
        output_features: int = 1024,
        dropout: float = 0.5,
        blend_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.layers = nn.ModuleList(
            CANetLayer(seq_len, patch_size, embed_dim, dropout, blend_ratio) for patch_size in patch_sizes
        )
        total_patches = sum(layer.patchifier.num_patches for layer in self.layers)
        self.fc1 = nn.Linear(embed_dim * total_patches, output_features)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_features, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        batch, _, m = x_enc.shape
        mean = x_enc.mean(dim=1, keepdim=True)
        std = x_enc.std(dim=1, keepdim=True)
        normalized = (x_enc - mean) / (std + 1e-5)

        branch_outputs = []
        for layer in self.layers:
            out = layer(normalized, mean, std)
            branch_outputs.append(out.reshape(batch * m, -1))
        concatenated = torch.cat(branch_outputs, dim=1)

        hidden = self.relu(self.drop(self.fc1(concatenated)))
        forecast = self.fc2(hidden).reshape(batch, m, self.pred_len).transpose(1, 2)
        return forecast * std + mean

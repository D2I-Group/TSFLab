"""Clean-room Dual-Path Wavelet Mixer (arXiv:2512.02070)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.wavelet import DecimatedWaveletTransform
from tsflab.models._inputs import check_window


class DualPathTrendMixer(nn.Module):
    """One resolution's forecaster: a rigid global-linear trend path fused
    with a flexible patch-embedding MLP-mixer path for local nonlinear
    evolution. The two paths are combined with unconstrained learnable
    scalars (not a softmax gate), matching the paper's fixed-but-learned
    fusion.
    """

    def __init__(
        self,
        length: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.global_linear = nn.Linear(length, pred_len)

        self.patch_len = min(patch_len, length)
        self.stride = max(1, min(stride, self.patch_len))
        self.num_patches = max(1, (length - self.patch_len) // self.stride + 1)
        self.needed_len = (self.num_patches - 1) * self.stride + self.patch_len

        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        d_ff = d_model * 2
        self.local_mixer = nn.Sequential(
            nn.Linear(d_model * self.num_patches, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len),
        )
        self.dropout = nn.Dropout(dropout)
        self.path_weight = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch*channels, length)
        global_out = self.global_linear(x)

        padded = x
        if padded.shape[-1] < self.needed_len:
            padded = F.pad(padded, (0, self.needed_len - padded.shape[-1]))
        patches = padded.unfold(-1, self.patch_len, self.stride)[:, : self.num_patches, :]
        tokens = self.dropout(self.patch_embedding(patches) + self.pos_embedding)
        local_out = self.local_mixer(tokens.reshape(tokens.shape[0], -1))

        return self.path_weight[0] * global_out + self.path_weight[1] * local_out


class Model(nn.Module):
    """DPWMixer: a lossless Haar wavelet pyramid replaces average-pooling
    down-sampling, and every resolution is forecast by an independent
    :class:`DualPathTrendMixer`; the per-resolution, per-channel forecasts
    are fused with a learned softmax weighting (Eq. "Adaptive Multi-Scale
    Fusion" of the paper).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        down_sampling_layers: int = 2,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.down_sampling_layers = down_sampling_layers

        self.revin = RevIN(enc_in)
        self.wavelets = nn.ModuleList(DecimatedWaveletTransform("haar", level=1) for _ in range(down_sampling_layers))

        lengths = [seq_len]
        current = seq_len
        for _ in range(down_sampling_layers):
            current = max(1, math.ceil(current / 2))
            lengths.append(current)

        self.mixers = nn.ModuleList(
            DualPathTrendMixer(length, pred_len, patch_len, stride, d_model, dropout) for length in lengths
        )
        self.fusion_weight = nn.Parameter(torch.ones(len(lengths), enc_in))
        self.last_fusion_weights: torch.Tensor | None = None

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]

        normalized = self.revin(x_enc, "norm").transpose(1, 2)  # (B, C, L)
        scales = [normalized]
        current = normalized
        for wavelet in self.wavelets:
            approx, _detail = wavelet.decompose(current)
            current = approx
            scales.append(current)

        outputs = []
        for mixer, signal in zip(self.mixers, scales):
            flat = signal.reshape(batch * self.enc_in, signal.shape[-1])
            outputs.append(mixer(flat).reshape(batch, self.enc_in, self.pred_len))
        stacked = torch.stack(outputs, dim=0)  # (S, B, C, P)

        weights = self.fusion_weight.softmax(dim=0)  # (S, C)
        self.last_fusion_weights = weights
        fused = (stacked * weights.unsqueeze(1).unsqueeze(-1)).sum(0)  # (B, C, P)

        return self.revin(fused.transpose(1, 2), "denorm")

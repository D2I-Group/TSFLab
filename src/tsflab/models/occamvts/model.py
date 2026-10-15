"""Clean-room OccamVTS deployment student from the public paper.

OccamVTS trains a compact temporal/visual student from a privileged vision
teacher. This module implements the retained inference-time student: overlapping
patch tokens, raw/frequency/periodic visual augmentation, a compact convolutional
visual encoder, and temporal-query cross-modal fusion (equations 1--2, 8--9, 12).
It intentionally excludes the training-only large vision teacher and distillation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        patch_len: int = 16,
        stride: int = 8,
        period: int = 24,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len = min(patch_len, seq_len)
        self.stride = min(stride, self.patch_len)
        self.num_patches = 1 + (seq_len - self.patch_len) // self.stride
        self.period = period
        self.revin = RevIN(enc_in, enabled=use_revin)
        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, 2 * d_model, dropout=dropout, batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.visual_encoder = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cross_modal = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.forecast = nn.Linear(d_model, pred_len)

    def visual_augmentation(self, history: torch.Tensor) -> torch.Tensor:
        """Raw, normalized FFT magnitude, and sine/cosine period channels."""
        spectrum = torch.fft.rfft(history, dim=-1).abs()
        magnitude = F.interpolate(spectrum, size=self.seq_len, mode="linear", align_corners=False)
        magnitude = magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        steps = torch.arange(self.seq_len, device=history.device, dtype=history.dtype)
        angle = 2 * math.pi * steps / self.period
        sine = angle.sin().reshape(1, 1, -1).expand_as(history)
        cosine = angle.cos().reshape(1, 1, -1).expand_as(history)
        return torch.stack((history, magnitude, sine, cosine), dim=2)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        history = normalized.transpose(1, 2)
        patches = history.unfold(-1, self.patch_len, self.stride)
        temporal = self.patch_projection(patches).reshape(-1, self.num_patches, self.patch_projection.out_features)
        temporal = self.temporal_encoder(temporal + self.position)

        augmented = self.visual_augmentation(history).reshape(-1, 4, self.seq_len)
        visual = self.visual_encoder(augmented)
        visual = F.adaptive_avg_pool1d(visual, self.num_patches).transpose(1, 2)
        attended, _ = self.cross_modal(temporal, visual, visual, need_weights=False)
        fused = self.fusion_norm(temporal + attended).mean(dim=1)
        forecast = self.forecast(fused).reshape(x_enc.shape[0], self.enc_in, self.pred_len)
        return self.revin(forecast.transpose(1, 2), "denorm")

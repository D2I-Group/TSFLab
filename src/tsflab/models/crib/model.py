"""Clean-room CRIB for forecasting directly from partially observed series.

The implementation follows the paper's patch embedding, unified-variate
attention, Gaussian information bottleneck, and augmented-view consistency
objective.  Missing entries can be supplied as NaNs or through ``mask``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


def observed_statistics(values: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample/channel statistics over observed values only."""
    weights = observed.to(values.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=1, keepdim=True) / count
    variance = ((values - mean).square() * weights).sum(dim=1, keepdim=True) / count
    return mean.detach(), torch.sqrt(variance + 1e-5).detach()


def temporal_encoding(length: int, width: int) -> torch.Tensor:
    """Sinusoidal temporal embedding from paper Eq. 3."""
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class PatchEmbedding(nn.Module):
    """Two-layer temporal convolution over value/missingness patch pairs."""

    def __init__(self, patch_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.network = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        patches = length // self.patch_len
        value_patches = values.transpose(1, 2).reshape(batch, channels, patches, self.patch_len)
        mask_patches = observed.transpose(1, 2).reshape_as(value_patches)
        pair = torch.stack((value_patches, mask_patches.to(values.dtype)), dim=3)
        pair = pair.reshape(batch * channels * patches, 2, self.patch_len)
        encoded = self.network(pair).mean(dim=-1)
        return encoded.reshape(batch, channels, patches, -1)


class UnifiedVariateEncoder(nn.Module):
    """Flatten all channel/patch tokens before standard self-attention (Eq. 4)."""

    def __init__(
        self,
        channels: int,
        patches: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.variable_embedding = nn.Parameter(torch.empty(channels, d_model))
        nn.init.normal_(self.variable_embedding, std=0.02)
        self.register_buffer("time_embedding", temporal_encoding(patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.variable_embedding[None, :, None, :]
        tokens = tokens + self.time_embedding[None, None, :, :]
        flattened = tokens.flatten(1, 2)
        return self.norm(self.encoder(flattened))


class Model(nn.Module):
    """Consistency-Regularized Information Bottleneck forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 8,
        model_dim: int = 32,
        heads_num: int = 4,
        enc_num: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        consis_weight: float = 1.0,
        kl_weight: float = 1e-6,
        augmentation_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len % patch_len:
            raise ValueError("patch_len must divide seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.patch_len = patch_len
        self.patches = seq_len // patch_len
        self.consis_weight = consis_weight
        self.kl_weight = kl_weight
        self.augmentation_rate = augmentation_rate
        self.patch_embedding = PatchEmbedding(patch_len, model_dim, dropout)
        self.unified_attention = UnifiedVariateEncoder(
            enc_in, self.patches, model_dim, heads_num, enc_num, dropout, activation
        )
        self.location = nn.Linear(model_dim, model_dim)
        self.log_scale = nn.Linear(model_dim, model_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.patches * model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, pred_len),
        )
        self.aux_loss: torch.Tensor | None = None

    def _encode(self, values: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embedding(values, observed)
        refined = self.unified_attention(tokens)
        return self.location(refined), self.log_scale(refined).clamp(-8.0, 8.0)

    def _augment(self, values: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keep = torch.rand_like(values) >= self.augmentation_rate
        augmented_observed = observed & keep
        noise = torch.randn_like(values) * 0.01
        augmented = torch.where(augmented_observed, values + noise, torch.zeros_like(values))
        return augmented, augmented_observed

    def forecast_masked(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.channels)
        finite = torch.isfinite(x_enc)
        if mask is not None:
            if mask.shape != x_enc.shape:
                raise ValueError("CRIB mask must have the same shape as x_enc")
            finite = finite & mask.to(torch.bool)
        safe = torch.where(finite, x_enc, torch.zeros_like(x_enc))
        mean, stdev = observed_statistics(safe, finite)
        normalized = torch.where(finite, (safe - mean) / stdev, torch.zeros_like(safe))
        location, log_scale = self._encode(normalized, finite)
        if self.training:
            latent = location + log_scale.exp() * torch.randn_like(location)
            augmented, augmented_observed = self._augment(normalized, finite)
            augmented_location, _ = self._encode(augmented, augmented_observed)
            compactness = 0.5 * (location.square() + torch.exp(2.0 * log_scale) - 1.0 - 2.0 * log_scale).mean()
            consistency = F.mse_loss(location, augmented_location)
            self.aux_loss = self.kl_weight * compactness + self.consis_weight * consistency
        else:
            latent = location
            self.aux_loss = None
        latent = latent.reshape(x_enc.shape[0], self.channels, self.patches, -1)
        forecast = self.predictor(latent.flatten(2)).transpose(1, 2)
        return forecast * stdev + mean

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        return self.forecast_masked(x_enc)

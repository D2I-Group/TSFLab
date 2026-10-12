"""Clean-room LSINet implementation.

LSINet (Zhang, Wang, Wang, and Wang, AAAI 2025) replaces the self-attention
scores of transformer forecasters with a *shared, sparse, sample-independent*
connection matrix over time patches (the Multihead Sparse Interaction
Mechanism, MSIM) learned through Shared Interaction Learning (SIL): one
connection matrix per head is generated from a learnable memory table rather
than from the input, so it is reused across every sample and channel.

Pipeline (paper Fig. 3): RevIN normalize -> patchify + learnable position
embedding (Patch Encoding) -> repeat(e_layers) Sparse Temporal Interaction
(STI) blocks, each composed of a time-invariant patch-mixing MLP, the shared
sparse connection routing + multi-head propagation (MSIM), a time-updating
MLP, and a feature-integration MLP with residual connections -> flatten +
linear forecast head -> RevIN denormalize.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN
from tsflab.models._components.sparse_connection_router import SharedSparseConnectionRouter


def _derive_patch_geometry(seq_len: int, n_patches: int) -> tuple[int, int, int]:
    """Derive ``(patch_len, stride, num_patches)`` from a target patch count.

    Follows the paper's Eq. 9 rule ``stride = floor(seq_len / n_patches)``,
    ``patch_len = 2 * stride``, then the standard end-replication-padded patch
    count of Eq. 1: ``num_patches = floor((seq_len - patch_len) / stride) + 2``.
    """
    stride = max(1, seq_len // max(n_patches, 1))
    patch_len = 2 * stride
    num_patches = (seq_len - patch_len) // stride + 2
    return patch_len, stride, num_patches


class _PatchMixMLP(nn.Module):
    """Two-layer MLP applied along the patch axis (Time-Invariant / Time-Updating)."""

    def __init__(self, num_patches: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches, num_patches),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch*channels, num_patches, d_model)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class _FeatureIntegrationMLP(nn.Module):
    """Two-layer MLP applied along the feature axis (Integration)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchEncoder(nn.Module):
    """Instance-normalized patching, linear projection, and learnable position embedding."""

    def __init__(
        self,
        patch_len: int,
        stride: int,
        num_patches: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad = nn.ReplicationPad1d((0, stride))
        self.project = nn.Linear(patch_len, d_model)
        self.position = positional_encoding("zeros", True, num_patches, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        x = self.pad(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.project(x) + self.position
        return self.dropout(x)


class SparseTemporalInteractionBlock(nn.Module):
    """One STI block: time-invariant mixing, sparse MSIM propagation, updating, integration."""

    def __init__(
        self,
        num_patches: int,
        d_model: int,
        n_heads: int,
        density: float,
        dropout: float,
    ) -> None:
        super().__init__()
        d_v = d_model // n_heads
        self.n_heads = n_heads
        self.d_v = d_v
        self.time_invariant = _PatchMixMLP(num_patches, dropout)
        self.router = SharedSparseConnectionRouter(
            num_positions=num_patches, dim=d_model, heads=n_heads, density=density
        )
        self.to_values = nn.Sequential(nn.Linear(d_model, d_v * n_heads), nn.Dropout(dropout))
        self.linear_align = nn.Sequential(nn.Linear(d_v * n_heads, d_model), nn.Dropout(dropout))
        self.time_updating = _PatchMixMLP(num_patches, dropout)
        self.integrate = _FeatureIntegrationMLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch*channels, num_patches, d_model)
        residual_block_input = x
        x_invariant = self.time_invariant(x)

        connections, _probabilities = self.router()  # (n_heads, N, N), shared across batch/channels
        batch = x.shape[0]
        values = self.to_values(x_invariant).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2)
        propagated = torch.matmul(connections.unsqueeze(0), values)  # (batch, heads, N, d_v)
        propagated = propagated.transpose(1, 2).reshape(batch, -1, self.n_heads * self.d_v)
        propagated = self.linear_align(propagated)

        x = propagated + residual_block_input
        x = self.time_updating(x)
        residual_after_update = x
        x = self.integrate(x) + residual_after_update + residual_block_input
        return x


class Model(nn.Module):
    """LSINet: RevIN -> patch encoding -> stacked STI (MSIM) blocks -> linear forecast head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 1,
        n_patches: int = 64,
        density: float = 0.15,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        patch_len, stride, num_patches = _derive_patch_geometry(seq_len, n_patches)
        self.revin = RevIN(enc_in, enabled=use_norm)
        self.patch_encoder = PatchEncoder(patch_len, stride, num_patches, d_model, dropout)
        self.blocks = nn.ModuleList(
            SparseTemporalInteractionBlock(num_patches, d_model, n_heads, density, dropout)
            for _ in range(e_layers)
        )
        self.head = FlattenForecastHead(
            individual=False,
            n_vars=enc_in,
            nf=d_model * num_patches,
            target_window=pred_len,
            head_dropout=dropout,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: (batch, seq_len, enc_in)
        batch, _, channels = x_enc.shape
        z = self.revin(x_enc, "norm")
        z = z.transpose(1, 2)  # (batch, channels, seq_len)

        patches = self.patch_encoder(z)  # (batch, channels, num_patches, d_model)
        num_patches = patches.shape[2]
        d_model = patches.shape[-1]
        flat = patches.reshape(batch * channels, num_patches, d_model)
        for block in self.blocks:
            flat = block(flat)
        flat = flat.reshape(batch, channels, num_patches, d_model).permute(0, 1, 3, 2)

        forecast = self.head(flat)  # (batch, channels, pred_len)
        forecast = forecast.transpose(1, 2)  # (batch, pred_len, channels)
        forecast = self.revin(forecast, "denorm")
        return forecast

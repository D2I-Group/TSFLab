"""STD-MAE forecasting stage: decoupled masked-autoencoder encoders plus Graph WaveNet.

Paper: Gao et al., "Spatio-Temporal-Decoupled Masked Pre-training for Spatiotemporal
Forecasting" (arXiv 2312.00516, IJCAI 2024).

Stage 1 (pre-training, NOT part of this module) trains two masked autoencoders on a
long history: a temporal one (T-MAE, attention along time within each node) and a
spatial one (S-MAE, attention across nodes within each time patch), each masking part
of its tokens and reconstructing them. Stage 2 (this module) freezes the two encoders,
encodes the unmasked long history, takes the last patch representation of every node
from both, and injects them into a Graph WaveNet that forecasts from the short history:

    h = [T-MAE(x_long)[:, :, -1], S-MAE(x_long)[:, :, -1]]            # (B, N, 2d)
    skip <- skip + MLP_t(h_t) + MLP_s(h_s)                            # before the head

The runner has no pre-training stage, so the encoders here are trained end to end
(default) or kept frozen with whatever weights were loaded into them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.diffusion_conv import DiffusionConv2d
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._components.tst_transformer import TSTEncoder
from tsflab.models._inputs import check_window, square_adjacency


def sincos_2d(rows: int, cols: int, width: int) -> torch.Tensor:
    """Sinusoidal 2-D table ``(rows, cols, width)``: row code then column code.

    Each axis uses ``ceil(width / 4) * 2`` channels of interleaved sin/cos over the
    frequencies ``10000 ** (-2k / channels)``; the two codes are concatenated and
    truncated to ``width``.
    """
    channels = int(math.ceil(width / 4) * 2)
    frequency = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def axis(length: int) -> torch.Tensor:
        angle = torch.arange(length).float()[:, None] * frequency[None, :]
        return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2, -1)

    table = torch.zeros(rows, cols, channels * 2)
    table[:, :, :channels] = axis(rows)[:, None, :]
    table[:, :, channels : 2 * channels] = axis(cols)[None, :, :]
    return table[:, :, :width]


class DecoupledMaskedEncoder(nn.Module):
    """Patch embedding and Transformer encoder of one masked autoencoder, unmasked.

    ``spatial=False`` attends along the patch axis within each node (T-MAE);
    ``spatial=True`` attends across nodes within each patch (S-MAE). Input
    ``(B, L, N)`` returns ``(B, N, P, d)`` with ``P = L // patch_size``.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        depth: int,
        dropout: float,
        spatial: bool,
    ) -> None:
        super().__init__()
        self.patch_size, self.embed_dim, self.spatial = patch_size, embed_dim, spatial
        self.num_patches = seq_len // patch_size
        self.patch_embedding = nn.Conv2d(1, embed_dim, kernel_size=(patch_size, 1), stride=(patch_size, 1))
        self.encoder = TSTEncoder(
            embed_dim,
            num_heads,
            n_layers=depth,
            d_ff=embed_dim * mlp_ratio,
            activation="relu",
            norm="LayerNorm",
            res_dropout=dropout,
        )
        self.register_buffer("position", sincos_2d(num_nodes, self.num_patches, embed_dim), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, nodes = values.shape
        series = values.transpose(1, 2).reshape(batch * nodes, 1, length, 1)
        patches = self.patch_embedding(series).squeeze(-1)  # (B*N, d, P)
        tokens = patches.view(batch, nodes, self.embed_dim, -1).transpose(-1, -2)
        tokens = tokens + self.position  # (B, N, P, d)
        if self.spatial:
            tokens = tokens.transpose(1, 2)  # (B, P, N, d): attend across nodes
        shape = tokens.shape
        scaled = tokens.reshape(-1, shape[2], self.embed_dim) * math.sqrt(self.embed_dim)
        encoded = self.encoder(scaled).view(shape)
        return encoded.transpose(1, 2) if self.spatial else encoded


class WaveNetGraphBlock(nn.Module):
    """Valid dilated gated convolution, diffusion graph convolution, residual, norm."""

    def __init__(
        self,
        residual: int,
        dilation_channels: int,
        skip: int,
        kernel: int,
        dilation: int,
        dropout: float,
        support_len: int,
    ) -> None:
        super().__init__()
        self.filter = nn.Conv2d(residual, dilation_channels, (1, kernel), dilation=(1, dilation))
        self.gate = nn.Conv2d(residual, dilation_channels, (1, kernel), dilation=(1, dilation))
        self.skip = nn.Conv2d(dilation_channels, skip, 1)
        self.graph = DiffusionConv2d(dilation_channels, residual, dropout, support_len, order=2)
        self.norm = nn.BatchNorm2d(residual)

    def forward(self, x, supports):
        gated = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        skip = self.skip(gated)
        mixed = self.graph(gated, supports) + x[..., -gated.shape[-1] :]
        return self.norm(mixed), skip


def _check_sizes(seq_len: int, num_nodes: int, history_len: int, patch_size: int) -> None:
    if num_nodes < 1:
        raise ValueError("STDMAE dimensions must be positive")
    if seq_len % patch_size:
        raise ValueError("seq_len must be a multiple of patch_size")
    if history_len > seq_len:
        raise ValueError("history_len must not exceed seq_len")


def receptive_field(blocks: int, layers: int, kernel_size: int) -> int:
    """Receptive field of the Graph WaveNet backend (``blocks`` x ``layers`` dilated convolutions)."""
    receptive = 1
    for _ in range(blocks):
        step = kernel_size - 1
        for _ in range(layers):
            receptive += step
            step *= 2
    return receptive


def _supports(adj_mx: np.ndarray | None, num_nodes: int):
    """Forward and reverse supports of ``adj_mx`` (the identity graph when absent)."""
    adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, num_nodes)
    return adj_to_supports(adjacency)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        history_len: int = 12,
        patch_size: int = 12,
        embed_dim: int = 96,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        encoder_depth: int = 4,
        encoder_dropout: float = 0.1,
        in_dim: int = 2,
        dropout: float = 0.3,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        freeze_encoders: bool = False,
    ) -> None:
        super().__init__()
        _check_sizes(seq_len, num_nodes, history_len, patch_size)
        receptive = receptive_field(blocks, layers, kernel_size)
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.history_len, self.in_dim, self.receptive_field = history_len, in_dim, receptive
        self.freeze_encoders = freeze_encoders

        self.temporal_encoder = DecoupledMaskedEncoder(
            num_nodes,
            seq_len,
            patch_size,
            embed_dim,
            num_heads,
            mlp_ratio,
            encoder_depth,
            encoder_dropout,
            spatial=False,
        )
        self.spatial_encoder = DecoupledMaskedEncoder(
            num_nodes,
            seq_len,
            patch_size,
            embed_dim,
            num_heads,
            mlp_ratio,
            encoder_depth,
            encoder_dropout,
            spatial=True,
        )
        if freeze_encoders:
            for encoder in (self.temporal_encoder, self.spatial_encoder):
                encoder.requires_grad_(False)

        forward_support, reverse_support = _supports(adj_mx, num_nodes)
        self.register_buffer("forward_support", forward_support)
        self.register_buffer("reverse_support", reverse_support)
        self.source_nodes = nn.Parameter(torch.randn(num_nodes, 10))
        self.target_nodes = nn.Parameter(torch.randn(10, num_nodes))

        self.start_conv = nn.Conv2d(in_dim, residual_channels, 1)
        self.blocks = nn.ModuleList(
            WaveNetGraphBlock(
                residual_channels,
                dilation_channels,
                skip_channels,
                kernel_size,
                2**layer,
                dropout,
                support_len=3,
            )
            for _ in range(blocks)
            for layer in range(layers)
        )
        self.temporal_context = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(), nn.Linear(512, skip_channels), nn.ReLU()
        )
        self.spatial_context = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(), nn.Linear(512, skip_channels), nn.ReLU()
        )
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, 1)
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:  # pre-trained encoders stay deterministic
            self.temporal_encoder.eval()
            self.spatial_encoder.eval()
        return self

    def encode_long_history(self, long_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Last-patch representations ``(B, N, d)`` from the temporal and spatial encoders."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoders):
            temporal = self.temporal_encoder(long_history)[:, :, -1]
            spatial = self.spatial_encoder(long_history)[:, :, -1]
        return temporal, spatial

    def _short_input(self, data: torch.Tensor) -> torch.Tensor:
        """Short history ``(B, C, N, L + 1)``, left-padded to at least the receptive field."""
        short = data[:, -self.history_len :, :, : self.in_dim]  # (B, L, N, C)
        x = F.pad(short.permute(0, 3, 2, 1), (1, 0, 0, 0))  # (B, C, N, L + 1)
        if x.shape[-1] < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.shape[-1], 0, 0, 0))
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.num_nodes)
        data = to_spatiotemporal(x_enc, x_mark_enc)
        if data.shape[-1] < self.in_dim:
            raise ValueError("STDMAE received fewer input features than configured")
        temporal, spatial = self.encode_long_history(x_enc)

        x = self.start_conv(self._short_input(data))
        supports = [
            self.forward_support,
            self.reverse_support,
            adaptive_node_embedding_adjacency(self.source_nodes, self.target_nodes),
        ]
        skip = None
        for block in self.blocks:
            x, contribution = block(x, supports)
            skip = contribution if skip is None else skip[..., -contribution.shape[-1] :] + contribution
        skip = skip + self.temporal_context(temporal).transpose(1, 2).unsqueeze(-1)
        skip = skip + self.spatial_context(spatial).transpose(1, 2).unsqueeze(-1)
        out = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))  # (B, P, N, 1)
        return out.squeeze(-1)

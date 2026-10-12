"""Clean-room PatchTST implementation of patching and channel independence."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN


def patchify(
    values: torch.Tensor,
    patch_len: int,
    stride: int,
    padding_patch: str,
) -> torch.Tensor:
    """Segment ``(B,L,C)`` into overlapping ``(B,C,N,P)`` patch tokens."""
    if values.ndim != 3:
        raise ValueError("patchify expects (batch, time, channels)")
    channel_first = values.transpose(1, 2)
    if padding_patch == "end":
        channel_first = F.pad(channel_first, (0, stride), mode="replicate")
    if channel_first.shape[-1] < patch_len:
        raise ValueError("patch_len exceeds the padded context window")
    return channel_first.unfold(-1, patch_len, stride)


def _position_encoding(length: int, width: int, kind: str) -> torch.Tensor:
    if kind == "zeros":
        return torch.zeros(1, length, width)
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class PatchEncoderLayer(nn.Module):
    """Shared Transformer block applied independently to every channel."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, activation: str, norm: str, attn_dropout: float, ffn_dropout: float, res_dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        nonlinear: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nonlinear,
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ff, d_model),
        )
        if norm == "LayerNorm":
            self.norm_one: nn.Module = nn.LayerNorm(d_model)
            self.norm_two: nn.Module = nn.LayerNorm(d_model)
        else:
            self.norm_one = nn.BatchNorm1d(d_model)
            self.norm_two = nn.BatchNorm1d(d_model)
        self.residual_dropout = nn.Dropout(res_dropout)

    @staticmethod
    def _normalize(norm: nn.Module, values: torch.Tensor) -> torch.Tensor:
        if isinstance(norm, nn.BatchNorm1d):
            return norm(values.transpose(1, 2)).transpose(1, 2)
        return norm(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        source = self._normalize(self.norm_one, values) if self.pre_norm else values
        attended, _ = self.attention(source, source, source, need_weights=False)
        values = values + self.residual_dropout(attended)
        if not self.pre_norm:
            values = self._normalize(self.norm_one, values)
        source = self._normalize(self.norm_two, values) if self.pre_norm else values
        values = values + self.residual_dropout(self.feed_forward(source))
        if not self.pre_norm:
            values = self._normalize(self.norm_two, values)
        return values


class Model(nn.Module):
    """Supervised PatchTST forecaster from the ICLR 2023 architecture."""

    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str = "end",
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        d_k: int | None = None,
        d_v: int | None = None,
        d_ff: int = 256,
        activation: str = "gelu",
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        head_dropout: float = 0.0,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        if min(context_window, target_window) < 1:
            raise ValueError("window values must be positive")
        probe = torch.zeros(1, context_window, c_in)
        patch_count = patchify(probe, patch_len, stride, padding_patch).shape[2]
        self.channels = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_count = patch_count
        self.revin = RevIN(
            c_in,
            affine=affine,
            subtract_last=subtract_last,
            enabled=revin,
        )
        self.patch_projection = nn.Linear(patch_len, d_model)
        position = _position_encoding(patch_count, d_model, pe)
        if learn_pe:
            self.position = nn.Parameter(position)
        else:
            self.register_buffer("position", position, persistent=True)
        self.projection_dropout = nn.Dropout(proj_dropout)
        self.encoder_layers = nn.ModuleList(
            [
                PatchEncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    activation,
                    norm,
                    attn_dropout,
                    ffn_dropout,
                    res_dropout,
                    pre_norm,
                )
                for _ in range(n_layers)
            ]
        )
        head_input = patch_count * d_model
        self.individual = individual
        if individual:
            self.head: nn.Module = nn.ModuleList(
                [nn.Linear(head_input, target_window) for _ in range(c_in)]
            )
        else:
            self.head = nn.Linear(head_input, target_window)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (
            self.context_window,
            self.channels,
        ):
            raise ValueError(
                f"x_enc must have shape (batch, {self.context_window}, {self.channels})"
            )
        normalized = self.revin(x_enc, "norm")
        patches = patchify(
            normalized, self.patch_len, self.stride, self.padding_patch
        )
        batch = patches.shape[0]
        tokens = self.patch_projection(patches).reshape(
            batch * self.channels, self.patch_count, -1
        )
        tokens = self.projection_dropout(tokens + self.position)
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        tokens = tokens.reshape(batch, self.channels, self.patch_count, -1)
        flattened = self.head_dropout(tokens.flatten(start_dim=2))
        if self.individual:
            predictions = torch.stack(
                [self.head[channel](flattened[:, channel]) for channel in range(self.channels)],
                dim=-1,
            )
        else:
            predictions = self.head(flattened).transpose(1, 2)
        return self.revin(predictions, "denorm")

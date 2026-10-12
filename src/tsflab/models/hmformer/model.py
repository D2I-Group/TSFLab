"""Clean-room HMformer with hierarchical multi-scale Transformer branches."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


def _absolute_position(length: int, dimension: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(dimension, 1))
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * frequency)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x[..., 0::2], x[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotarySelfAttention(nn.Module):
    """Scaled dot-product attention with rotary temporal position encoding."""

    def __init__(self, dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def _rope(self, tensor: torch.Tensor) -> torch.Tensor:
        length = tensor.shape[-2]
        position = torch.arange(length, device=tensor.device, dtype=tensor.dtype)
        inverse = torch.exp(
            torch.arange(0, self.head_dim, 2, device=tensor.device, dtype=tensor.dtype)
            * (-math.log(10000.0) / self.head_dim)
        )
        phase = torch.outer(position, inverse).repeat_interleave(2, dim=-1)
        return tensor * phase.cos()[None, None] + _rotate_half(tensor) * phase.sin()[None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = self._rope(query), self._rope(key)
        weights = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        attended = self.dropout(weights) @ value
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        return self.output(attended)


class TransformerBlock(nn.Module):
    """The post-normalized attention and feed-forward block in Eq. (3)."""

    def __init__(self, dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = RotarySelfAttention(dimension, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x + self.dropout(self.attention(x)))
        return self.feed_forward_norm(x + self.dropout(self.feed_forward(x)))


class ScaleBranch(nn.Module):
    """One SAFE branch: patch embedding, Transformer blocks, and flat head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        dimension: int,
        num_heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = max(1, (seq_len - patch_len) // stride + 2)
        self.embedding = nn.Conv1d(patch_len, dimension, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dimension, num_heads, dropout) for _ in range(depth)]
        )
        self.predictor = nn.Linear(self.num_patches * dimension, pred_len)

    def patch(self, series: torch.Tensor) -> torch.Tensor:
        padded = F.pad(series, (0, self.stride), mode="replicate")
        if padded.shape[-1] < self.patch_len:
            padded = F.pad(padded, (0, self.patch_len - padded.shape[-1]), mode="replicate")
        patches = padded.unfold(-1, self.patch_len, self.stride)
        if patches.shape[1] != self.num_patches:
            patches = F.interpolate(
                patches.transpose(1, 2), size=self.num_patches, mode="linear", align_corners=False
            ).transpose(1, 2)
        return patches

    def embed(self, series: torch.Tensor) -> torch.Tensor:
        patches = self.patch(series)
        tokens = self.embedding(patches.transpose(1, 2)).transpose(1, 2)
        return tokens + _absolute_position(
            tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype
        ).unsqueeze(0)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class Model(nn.Module):
    """Hierarchical cross-scale mixing with complementary branch forecasts."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        num_scales: int = 3,
        depth: int = 1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        effective_scales = 1
        while effective_scales < num_scales and patch_len * 2**effective_scales <= seq_len:
            effective_scales += 1
        self.num_scales = effective_scales
        branches: list[ScaleBranch] = []
        for scale in range(effective_scales):
            dimension = d_model * 2**scale
            branches.append(
                ScaleBranch(
                    seq_len,
                    pred_len,
                    patch_len * 2**scale,
                    stride * 2**scale,
                    dimension,
                    num_heads,
                    depth,
                    dropout,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.cross_scale = nn.ModuleList(
            [
                nn.Conv1d(d_model * 2**scale, d_model * 2 ** (scale + 1), 2, stride=2)
                for scale in range(effective_scales - 1)
            ]
        )

    def branch_representations(self, x: torch.Tensor) -> list[torch.Tensor]:
        check_window(x, self.seq_len, self.enc_in)
        batch = x.shape[0]
        series = x.transpose(1, 2).reshape(batch * self.enc_in, self.seq_len)
        outputs: list[torch.Tensor] = []
        mixed: torch.Tensor | None = None
        for index, branch in enumerate(self.branches):
            tokens = branch.embed(series)
            if mixed is not None:
                if mixed.shape[1] != tokens.shape[1]:
                    mixed = F.interpolate(
                        mixed.transpose(1, 2),
                        size=tokens.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                tokens = tokens + mixed
            tokens = branch.encode(tokens)
            outputs.append(tokens)
            if index < len(self.cross_scale):
                mixed = self.cross_scale[index](tokens.transpose(1, 2)).transpose(1, 2)
        return outputs

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        batch = x_enc.shape[0]
        outputs = self.branch_representations(x_enc)
        forecasts = [
            branch.predictor(tokens.flatten(1))
            for branch, tokens in zip(self.branches, outputs, strict=True)
        ]
        forecast = torch.stack(forecasts).sum(0)
        return forecast.reshape(batch, self.enc_in, self.pred_len).transpose(1, 2)

"""Local SCFormer implementation.

Independent rewrite of Section 3 (Eqs. 1-16) of Guo et al., arXiv 2505.02655,
with omissions resolved by reading the official repository
(``ShiweiGuo1995/SCFormer``, revision ``a6f5aec``; the model is named
``SiTransformer`` there). Nothing was copied.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import solve_triangular

from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.self_attention_family import FullAttention

StructMode = Literal["triangular", "conv"]

# Official convolutional mode: 3 stacked query/key/value convolutions with
# kernel ``kernel_size // 2`` and a fixed dropout of 0.1 before their LayerNorm.
_CONV_QKV_LAYERS = 3
_CONV_QKV_DROPOUT = 0.1


def legs_transition(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (1): continuous HiPPO-LegS ``(A, B)``.

    ``A[n, k] = -sqrt(2n+1) sqrt(2k+1)`` for ``n > k``, ``-(n+1)`` for ``n = k``,
    zero above the diagonal; ``B[n] = sqrt(2n+1)``.
    """
    n = np.arange(order, dtype=np.float64)
    root = np.sqrt(2.0 * n + 1.0)
    a = -np.tril(np.outer(root, root), -1) - np.diag(n + 1.0)
    return a, root


@lru_cache(maxsize=8)
def legs_projection(order: int, length: int) -> np.ndarray:
    """``[length, order]`` matrix ``W`` with ``c_length = x @ W`` (float64).

    The bilinear-discretized LegS recurrence ``c_t = A_t c_{t-1} + B_t x_t`` with
    ``A_t = (I - A/2t)^-1 (I + A/2t)``, ``B_t = (I - A/2t)^-1 B/t`` and ``c_0 = 0``
    is linear in the input, so the state after ``length`` steps is a fixed
    projection: ``W[k] = A_length ... A_{k+2} B_{k+1}`` (0-based row ``k``).
    """
    a, b = legs_transition(order)
    eye = np.eye(order)
    weights = np.empty((length, order))
    carry = eye  # product of the transitions applied after step k
    for k in range(length - 1, -1, -1):
        t = k + 1.0
        lhs = eye - a / (2.0 * t)
        b_t = solve_triangular(lhs, b / t, lower=True)
        weights[k] = carry @ b_t
        a_t = solve_triangular(lhs, eye + a / (2.0 * t), lower=True)
        carry = carry @ a_t
    return weights


def _conv_same_prefix(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """Apply a ``Conv1d(1, 1, k, padding=(k - 1) // 2 + 1)`` along the last axis, keep the first ``D``."""
    shape = x.shape
    out = conv(x.reshape(-1, 1, shape[-1]))[..., : shape[-1]]
    return out.reshape(shape)


def _conv(kernel: int) -> nn.Conv1d:
    return nn.Conv1d(1, 1, kernel, padding=(kernel - 1) // 2 + 1)


class StructuredLinear(nn.Module):
    """Eqs. (4) and (7): ``y = x @ M + b`` with ``M[i, j] = 0`` whenever ``i < j``.

    Output position ``j`` of the embedded series reads only positions ``i >= j``,
    so every structured map in the encoder keeps one temporal direction.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.register_buffer("mask", torch.tril(torch.ones(width, width)), persistent=False)

    def matrix(self) -> torch.Tensor:
        return self.linear.weight * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix() + self.linear.bias


class StructuredAttention(nn.Module):
    """Channel-wise self-attention with structured query/key/value/output maps.

    ``triangular``: Eqs. (4)-(7) with lower-triangular weight masks.
    ``conv``: Eqs. (11)-(14), stacked 1D convolutions along the embedded series
    followed by ReLU, dropout and LayerNorm for queries/keys/values.
    """

    def __init__(self, d_model: int, n_heads: int, mode: StructMode, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.mode = n_heads, mode
        self.attention = FullAttention(False, attention_dropout=dropout)
        if mode == "triangular":
            self.qkv = nn.ModuleList(StructuredLinear(d_model) for _ in range(3))
            self.out = StructuredLinear(d_model)
        else:
            half = kernel_size // 2
            self.qkv = nn.ModuleList(
                nn.ModuleList(_conv(half) for _ in range(_CONV_QKV_LAYERS)) for _ in range(3)
            )
            self.qkv_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(3))
            self.qkv_dropout = nn.Dropout(_CONV_QKV_DROPOUT)
            self.out = _conv(kernel_size)

    def project(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch, channels, _ = x.shape
        if self.mode == "triangular":
            projected = [layer(x) for layer in self.qkv]
        else:
            projected = []
            for stack, norm in zip(self.qkv, self.qkv_norms):
                y = x
                for conv in stack:
                    y = _conv_same_prefix(conv, y)
                projected.append(norm(self.qkv_dropout(F.relu(y))))
        return [p.reshape(batch, channels, self.n_heads, -1) for p in projected]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, width = x.shape
        queries, keys, values = self.project(x)
        out, _ = self.attention(queries, keys, values, None)
        out = out.reshape(batch, channels, width)
        if self.mode == "triangular":
            return self.out(out)
        return _conv_same_prefix(self.out, out)


class EncoderLayer(nn.Module):
    """Residual attention, LayerNorm, structured two-layer feed-forward, residual LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, mode: StructMode, kernel_size: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.attention = StructuredAttention(d_model, n_heads, mode, kernel_size, dropout)
        self.mode = mode
        if mode == "triangular":
            self.fc1, self.fc2 = StructuredLinear(d_model), StructuredLinear(d_model)
        else:
            self.fc1, self.fc2 = _conv(kernel_size), _conv(kernel_size)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def _map(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return layer(x) if self.mode == "triangular" else _conv_same_prefix(layer, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x))
        y = self.dropout(self.activation(self._map(self.fc1, self.norm1(x))))
        y = self.dropout(self._map(self.fc2, y))
        return self.norm2(x + y)


class Model(nn.Module):
    """SCFormer: HiPPO-LegS cumulative state + structured channel-wise Transformer.

    ``x_enc`` carries ``seq_len = history + lookback - 1`` steps: the last
    ``lookback`` steps are the look-back window; the first
    ``seq_len - lookback + 1`` steps (ending at the first look-back step) are the
    accumulated history summarized by the HiPPO state.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        lookback: int = 96,
        hippo_order: int = 512,
        d_model: int = 512,
        n_heads: int = 16,
        e_layers: int = 2,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
        struct_mode: StructMode = "triangular",
        kernel_size: int = 32,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("SCFormer sizes must be positive")
        if lookback > seq_len:
            raise ValueError("SCFormer needs lookback <= seq_len (the rest of the window is history)")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.lookback, self.history = lookback, seq_len - lookback + 1
        self.use_norm = use_norm
        projection = legs_projection(hippo_order, self.history)
        self.register_buffer("hippo_projection", torch.as_tensor(projection, dtype=torch.float32), persistent=False)
        # Eq. (2): Z = MLP(Concat([MLP(l), c])) plus a learnable per-channel embedding.
        self.window_proj = nn.Linear(lookback, d_model)
        self.embed = nn.Linear(hippo_order + d_model, d_model)
        self.channel_embedding = nn.Parameter(torch.empty(1, enc_in, d_model).uniform_(-0.02, 0.02))
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, struct_mode, kernel_size, dropout, activation) for _ in range(e_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len)

    def hippo_state(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Eq. (1): LegS coefficients of the history ending at the first look-back step, ``[B, C, N]``."""
        history = x_enc[:, : self.history, :]
        return torch.einsum("blc,ln->bcn", history, self.hippo_projection.to(history.dtype))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or tuple(x_enc.shape[1:]) != (self.seq_len, self.enc_in):
            raise ValueError(f"SCFormer expects (B, {self.seq_len}, {self.enc_in}) values")
        state = self.hippo_state(x_enc)  # computed on the loader-scaled values, before instance norm
        window = x_enc[:, -self.lookback :, :]
        if self.use_norm:  # Eq. (15)
            window, means, stdev = standardize(window)
        z = self.embed(torch.cat([self.window_proj(window.transpose(1, 2)), state], dim=-1))
        z = z + self.channel_embedding
        for layer in self.layers:
            z = layer(z)
        out = self.projector(self.norm(z)).transpose(1, 2)  # [B, pred_len, C]
        if self.use_norm:
            out = destandardize(out, means, stdev)
        return out


__all__ = [
    "EncoderLayer",
    "Model",
    "StructuredAttention",
    "StructuredLinear",
    "legs_projection",
    "legs_transition",
]

"""Local WaveRoRA: wavelet-domain variate tokens mixed by Rotary Route Attention.

Paper map (Liang, Sun and Guizani, arXiv 2410.22649): instance normalization
(Section IV-B); a J-level DWT whose J detail and one approximation series are each
embedded by their own linear layer (Section IV-C, Eq. 7); N WaveRoRA encoder
layers, each a Rotary Route Attention over the series-wise tokens (Section IV-D,
Alg. 1, Eq. 11) followed by wavelet-wise normalization blocks (Eq. 12); one
predictor per coefficient series producing the horizon's wavelet coefficients
(Section IV-E, Eq. 13); the inverse DWT (Eq. 3-4) and de-normalization.

Omissions of the paper are resolved from the pinned official code
(Leopold2333/WaveRoRA, ``models/WaveRoRA.py``, ``layers/WaveRoRA_layer.py``,
``layers/Attention.py``, ``layers/Transformer_EncDec.py``, ``utils/rotation.py``);
see the model card for the mapping and differences.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.orthogonal_dwt import OrthogonalDWT, coefficient_lengths
from tsflab.models._components.revin import RevIN

ROPE_BASE = 10000


def rotary_positions(scores: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs of the last axis by ``p * theta_k`` (Eq. 8).

    The rotated axis is the router axis ``r`` (``theta_k = 10000^(-2k/r)``); the
    position ``p`` of a row along the second-to-last (variate) axis is counted from
    the end, ``p = T - 1 - i``, as in the official reversed RoPE table.
    """
    rows, width = scores.shape[-2], scores.shape[-1]
    half = width // 2
    theta = ROPE_BASE ** (-torch.arange(half, device=scores.device, dtype=scores.dtype) / half)
    positions = torch.arange(rows - 1, -1, -1, device=scores.device, dtype=scores.dtype)
    angle = positions[:, None] * theta[None, :]
    cos, sin = torch.cos(angle), torch.sin(angle)
    pairs = scores.reshape(*scores.shape[:-1], half, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
    return rotated.flatten(-2)


class RotaryRouteAttention(nn.Module):
    """Gated multi-head Rotary Route Attention over ``[B, M, d_model]`` tokens (Alg. 1, Eq. 11).

    ``r`` learned routing tokens first attend to the keys (softmax over the
    variates) and gather values; every query then attends to the routing tokens
    (softmax over ``r``). Both score matrices are rotated along the router axis
    with position-dependent angles before use, so the product carries relative
    variate positions. A linear skip of the values and a SiLU gate complete the
    layer. Values are the input tokens themselves, split into heads.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        router_num: int,
        dropout: float,
        rotary: bool = True,
        residual: bool = True,
        gate: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.rotary = rotary
        self.residual = residual
        self.gate = gate
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.routers = nn.Parameter(torch.randn(router_num, d_model))
        self.router_dropout = nn.Dropout(dropout)
        self.query_dropout = nn.Dropout(dropout)
        if residual:
            self.skip = nn.Linear(d_model, d_model)
        if gate:
            self.gate_projection = nn.Linear(d_model, d_model)

    def attention(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, width = tokens.shape
        heads = self.n_heads
        q = self.query(tokens).view(batch, count, heads, -1).transpose(1, 2)
        k = self.key(tokens).view(batch, count, heads, -1).transpose(1, 2)
        v = tokens.view(batch, count, heads, -1).transpose(1, 2)
        r = self.routers.view(self.routers.shape[0], heads, -1).transpose(0, 1)
        scale = 1.0 / math.sqrt(k.shape[-1])
        # Routing tokens aggregate from all variates: [B, H, r, M].
        router_weights = torch.softmax(scale * torch.einsum("hre,bhse->bhrs", r, k), dim=-1)
        if self.rotary:
            router_weights = rotary_positions(router_weights.transpose(-1, -2)).transpose(-1, -2)
        router_values = self.router_dropout(router_weights) @ v
        # Each variate reads from the routing tokens: [B, H, M, r].
        query_weights = torch.softmax(scale * torch.einsum("bhle,hre->bhlr", q, r), dim=-1)
        if self.rotary:
            query_weights = rotary_positions(query_weights)
        out = self.query_dropout(query_weights) @ router_values
        return out.transpose(1, 2).reshape(batch, count, width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.attention(tokens)
        if self.residual:
            out = out + self.skip(tokens)
        if self.gate:
            out = out * F.silu(self.gate_projection(tokens))
        return out


class WaveNormBlock(nn.Module):
    """Post-norm residual add and convolutional feed-forward for one coefficient series (Eq. 12).

    ``x = LN(x + drop(u))``; ``LN(x + drop(conv2(drop(gelu(conv1(x))))))`` where the
    convolutions run along the variate axis with odd ``kernel_size`` (pointwise for 1).
    """

    def __init__(self, width: int, d_ff: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(width, d_ff, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(d_ff, width, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(update))
        y = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        return self.norm2(x + y)


class WaveRoRAEncoderLayer(nn.Module):
    """One encoder layer on wavelet-wise tokens ``[B, M, J + 1, D]``.

    Series-wise tokens (the ``J + 1`` embeddings of a variate concatenated) are
    projected to ``d_model``, mixed by Rotary Route Attention, projected back, split
    into wavelet-wise tokens, and each coefficient series passes its own
    ``WaveNormBlock`` with the layer input as residual.
    """

    def __init__(
        self,
        series: int,
        wavelet_dim: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        router_num: int,
        kernel_size: int,
        dropout: float,
        rotary: bool,
        residual: bool,
        gate: bool,
    ) -> None:
        super().__init__()
        self.expand = nn.Linear(series * wavelet_dim, d_model)
        self.attention = RotaryRouteAttention(d_model, n_heads, router_num, dropout, rotary, residual, gate)
        self.contract = nn.Linear(d_model, series * wavelet_dim)
        self.blocks = nn.ModuleList(WaveNormBlock(wavelet_dim, d_ff, kernel_size, dropout) for _ in range(series))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, variates, series, width = x.shape
        update = self.contract(self.attention(self.expand(x.reshape(batch, variates, series * width))))
        update = update.view(batch, variates, series, width)
        return torch.stack([block(x[:, :, j], update[:, :, j]) for j, block in enumerate(self.blocks)], dim=2)


def default_router_num(variates: int) -> int:
    """Official rule ``int(sqrt(M) + log2(M)) // 2`` rounded up to even (floored at 2)."""
    count = int(math.sqrt(variates) + math.log2(variates)) // 2
    return max(count + count % 2, 2)


class Model(nn.Module):
    """WaveRoRA forecaster on ``[B, L, M]`` histories."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        wavelet: str = "sym3",
        wavelet_layers: int = 3,
        wavelet_dim: int = 64,
        d_model: int = 256,
        d_ff: int = 256,
        n_heads: int = 8,
        e_layers: int = 1,
        router_num: int = 0,
        kernel_size: int = 1,
        dropout: float = 0.3,
        rotary: bool = True,
        residual: bool = True,
        gate: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        routers = router_num if router_num > 0 else default_router_num(enc_in)
        # RoPE rotates pairs along the router axis: an odd count is rounded up.
        self.router_num = routers + routers % 2

        self.revin = RevIN(enc_in, affine=False)
        # J-level DWT with zero extension (PyWavelets ``mode="zero"``); the filter
        # buffers stay in the state dict as ``dwt.analysis`` / ``dwt.synthesis``.
        self.dwt = OrthogonalDWT(wavelet, wavelet_layers, mode="zero")
        size = self.dwt.filter_length
        self.input_lengths = coefficient_lengths(seq_len, size, wavelet_layers)
        self.output_lengths = coefficient_lengths(pred_len, size, wavelet_layers)
        series = wavelet_layers + 1
        self.embeddings = nn.ModuleList(nn.Linear(length, wavelet_dim) for length in self.input_lengths)
        self.layers = nn.ModuleList(
            WaveRoRAEncoderLayer(
                series, wavelet_dim, d_model, d_ff, n_heads, self.router_num, kernel_size, dropout,
                rotary, residual, gate,
            )
            for _ in range(e_layers)
        )
        self.predictors = nn.ModuleList(
            nn.Sequential(nn.Linear(wavelet_dim, 2 * wavelet_dim), nn.GELU(), nn.Linear(2 * wavelet_dim, length))
            for length in self.output_lengths
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must be [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        normalized = self.revin(x_enc, "norm").transpose(1, 2)
        coefficients = self.dwt.decompose(normalized)
        tokens = torch.stack([embed(c) for embed, c in zip(self.embeddings, coefficients)], dim=2)
        for layer in self.layers:
            tokens = layer(tokens)
        predicted = [predict(tokens[:, :, j]) for j, predict in enumerate(self.predictors)]
        output = self.dwt.reconstruct(predicted)[..., : self.pred_len]
        return self.revin(output.transpose(1, 2), "denorm")

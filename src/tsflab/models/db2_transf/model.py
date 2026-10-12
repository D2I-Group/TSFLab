"""Local DB2-TransF: learnable Daubechies (db2) wavelet mixing over variate tokens.

The history of every variate (and, optionally, every TSLib ``timeF`` calendar
feature) is embedded as one token, as in inverted Transformers. Each encoder
block replaces self-attention with a multi-head, multi-level learnable db2
analysis along the token axis: per head, every level splits its input into
approximation and detail coefficients with learnable four-tap filters (paper
Eqs. 10-14); the final approximation and all details are linearly
interpolated back to the token count, summed, projected, and added residually.
A pre-norm feed-forward network and a per-block BatchNorm follow.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.marks import encoder_timef_marks
from tsflab.models._components.revin import RevIN

_SQRT3 = math.sqrt(3.0)
_NORM = 4.0 * math.sqrt(2.0)
#: db2 low-pass taps, paper Eq. (6).
DB2_LOW_PASS = ((1 + _SQRT3) / _NORM, (3 + _SQRT3) / _NORM, (3 - _SQRT3) / _NORM, (1 - _SQRT3) / _NORM)
#: High-pass taps used for initialisation. These are the negated taps of paper
#: Eq. (7); the sign follows the pinned official initialisation (see README).
DB2_HIGH_PASS_INIT = (
    (1 - _SQRT3) / _NORM,
    -(3 - _SQRT3) / _NORM,
    (3 + _SQRT3) / _NORM,
    -(1 + _SQRT3) / _NORM,
)


def pad_for_windows(x: torch.Tensor, taps: int = 4, stride: int = 2) -> torch.Tensor:
    """Right zero-pad ``[B, T, D]`` so that T >= taps and (T - taps) % stride == 0."""
    length = max(x.shape[1], taps)
    length += (length - taps) % stride
    if length == x.shape[1]:
        return x
    return F.pad(x, (0, 0, 0, length - x.shape[1]))


class LearnableDB2(nn.Module):
    """One analysis level, Eqs. (10)-(11): A[n] = sum_k alpha_k * x[2n+k], D[n] = sum_k beta_k * x[2n+k].

    ``alpha`` and ``beta`` are ``[4, dim]`` (one learnable tap per feature),
    initialised from the db2 taps.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(DB2_LOW_PASS).unsqueeze(1).repeat(1, dim))
        self.beta = nn.Parameter(torch.tensor(DB2_HIGH_PASS_INIT).unsqueeze(1).repeat(1, dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"expected [batch, tokens, {self.dim}], got {tuple(x.shape)}")
        windows = pad_for_windows(x).unfold(1, 4, 2)  # [B, n, D, 4]
        approx = torch.einsum("bndk,kd->bnd", windows, self.alpha)
        detail = torch.einsum("bndk,kd->bnd", windows, self.beta)
        return approx, detail


class MultiLevelDB2(nn.Module):
    """Eqs. (12)-(15): iterate the analysis on the approximation for ``levels`` levels."""

    def __init__(self, dim: int, levels: int) -> None:
        super().__init__()
        self.levels = nn.ModuleList(LearnableDB2(dim) for _ in range(levels))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        details = []
        approx = x
        for level in self.levels:
            approx, detail = level(approx)
            details.append(detail)
        return approx, details

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample A_L and D_1..D_L to the input token count (linear) and sum them."""
        approx, details = self(x)
        tokens = x.shape[1]
        out = torch.zeros_like(x)
        for part in (approx, *details):
            out = out + F.interpolate(
                part.transpose(1, 2), size=tokens, mode="linear", align_corners=False
            ).transpose(1, 2)
        return out


class DB2Block(nn.Module):
    """Multi-head multi-scale learnable-db2 block (MLDB) with a pre-norm FFN."""

    def __init__(self, d_model: int, d_ff: int, levels: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.heads = nn.ModuleList(MultiLevelDB2(self.head_dim, levels) for _ in range(n_heads))
        self.norm_mix = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm_mix(x)
        parts = normed.split(self.head_dim, dim=-1)
        mixed = torch.cat([head.mix(part) for head, part in zip(self.heads, parts)], dim=-1)
        x = x + self.projection(mixed)
        return x + self.ffn(self.norm_ffn(x))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 256,
        e_layers: int = 3,
        levels: int = 1,
        n_heads: int = 8,
        d_ff: int = 768,
        dropout: float = 0.1,
        use_norm: bool = True,
        use_marks: bool = True,
        freq: str = "h",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("DB2TransF sizes must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_norm = use_norm
        self.use_marks = use_marks
        self.freq = freq
        self.normalizer = RevIN(enc_in, eps=1e-5, affine=False, enabled=use_norm)
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.blocks = nn.ModuleList(DB2Block(d_model, d_ff, levels, n_heads) for _ in range(e_layers))
        self.block_norms = nn.ModuleList(nn.BatchNorm1d(d_model) for _ in range(e_layers))
        self.projector = nn.Linear(d_model, pred_len)

    def calendar_tokens(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        return encoder_timef_marks(x_mark_enc, seq_len=self.seq_len, freq=self.freq, enabled=self.use_marks)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        x = self.normalizer(x_enc, "norm")
        tokens = self.embedding(x, self.calendar_tokens(x_mark_enc))  # [B, N(+marks), d_model]
        for block, norm in zip(self.blocks, self.block_norms):
            tokens = norm(block(tokens).transpose(1, 2)).transpose(1, 2)
        forecast = self.projector(tokens).transpose(1, 2)[:, :, : self.enc_in]
        return self.normalizer(forecast, "denorm")

"""Clean-room WDformer (arXiv:2509.25231): wavelet-subband variate embedding,
an inverted (variate-token) differential-attention encoder, and a learned
linear map to pseudo-subband coefficients that a fixed inverse wavelet
transform reassembles into the forecast.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.differential_attention import DifferentialAttention
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.wavelet import DecimatedWaveletTransform
from tsflab.models._inputs import check_window


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with a learned per-feature scale."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    """Gated-linear feed-forward block with a SiLU-activated gate branch."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_model * 2)
        self.value = nn.Linear(d_model, d_model * 2)
        self.project = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.nn.functional.silu(self.gate(x)) * self.value(x))


class DifferentialSelfAttentionLayer(nn.Module):
    """Project to doubled query/key/value dimensions, apply differential
    attention, and project back to ``d_model``."""

    def __init__(self, d_model: int, n_heads: int, lambda_init: float, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, 2 * d_model)
        self.k_proj = nn.Linear(d_model, 2 * d_model)
        self.v_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)
        self.attention = DifferentialAttention(d_model, n_heads, lambda_init=lambda_init, attention_dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, tokens, self.n_heads, 2 * self.head_dim)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        out = self.attention(q, k, v).reshape(batch, tokens, -1)
        return self.out_proj(out)


class WDformerEncoderLayer(nn.Module):
    """Pre-norm differential self-attention residual, then a post-norm SwiGLU
    feed-forward residual (the feed-forward residual is added to the
    *normalized* attention output, matching the official encoder layer's
    ``x = norm(x); return norm(x + ff(x))`` pattern).
    """

    def __init__(self, d_model: int, n_heads: int, lambda_init: float, dropout: float) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attention = DifferentialSelfAttentionLayer(d_model, n_heads, lambda_init, dropout)
        self.dropout = nn.Dropout(dropout)
        self.mid_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLU(d_model)
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(x)
        x = x + self.dropout(self.attention(normed))
        x = self.mid_norm(x)
        return self.out_norm(x + self.feed_forward(x))


def _subband_sizes(wavelet: DecimatedWaveletTransform, length: int) -> list[int]:
    """Coefficient count of every subband that ``wavelet`` produces for a ``length``-step series."""
    with torch.no_grad():
        return [c.shape[-1] for c in wavelet.decompose(torch.zeros(1, 1, length))]


class Model(nn.Module):
    """WDformer: a Haar wavelet decomposition of the raw window feeds a
    per-subband linear embedding (concatenated into one ``d_model``-wide
    token per variate); an inverted-Transformer-style encoder of stacked
    differential-attention layers attends over variate tokens; and a linear
    projection produces pseudo-subband coefficients that a fixed inverse
    Haar wavelet transform reassembles into the pointwise forecast.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        wave_size: int = 3,
    ) -> None:
        super().__init__()
        del d_ff  # accepted for preset/CLI parity; the official SwiGLU feed-forward
        # width is fixed at 2 * d_model and never reads d_ff (its only official
        # consumer, a Conv1d-based feed-forward branch, is unused dead code in
        # the pinned official EncoderLayer.forward).

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.wave_size = wave_size

        self.embed_wavelet = DecimatedWaveletTransform("haar", level=wave_size)
        embed_sizes = _subband_sizes(self.embed_wavelet, seq_len)
        band_dim = d_model // (wave_size + 1)
        band_dims = [band_dim] * wave_size + [d_model - wave_size * band_dim]
        self.subband_embedding = nn.ModuleList(nn.Linear(size, dim) for size, dim in zip(embed_sizes, band_dims))
        self.embedding_dropout = nn.Dropout(dropout)

        lambda_schedule = [0.7 - 0.5 * math.exp(-0.3 * layer) for layer in range(e_layers)]
        self.layers = nn.ModuleList(
            WDformerEncoderLayer(d_model, n_heads, lambda_init, dropout) for lambda_init in lambda_schedule
        )
        self.final_norm = RMSNorm(d_model)

        self.output_wavelet = DecimatedWaveletTransform("haar", level=wave_size)
        self.output_split_sizes = _subband_sizes(self.output_wavelet, pred_len)
        # The deepest level may pad an odd intermediate length by one sample
        # (trimmed again inside ``reconstruct``), so the fabricated
        # coefficients can total one or two elements more than ``pred_len``.
        self.projector = nn.Linear(d_model, sum(self.output_split_sizes))

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.enc_in)

        normalized, mean, std = standardize(x_enc)

        variate_first = normalized.transpose(1, 2)  # (B, enc_in, seq_len)
        if x_mark_enc is not None:
            variate_first = torch.cat([variate_first, x_mark_enc.transpose(1, 2)], dim=1)

        subbands = self.embed_wavelet.decompose(variate_first)
        tokens = torch.cat([embed(subband) for embed, subband in zip(self.subband_embedding, subbands)], dim=-1)
        tokens = self.embedding_dropout(tokens)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)

        pseudo_subbands = self.projector(tokens)  # (B, enc_in[+marks], pred_len)
        chunks = list(torch.split(pseudo_subbands, self.output_split_sizes, dim=-1))
        reconstructed = self.output_wavelet.reconstruct(chunks)  # (B, enc_in[+marks], pred_len)
        forecast = reconstructed[:, : self.enc_in, :].transpose(1, 2)

        return destandardize(forecast, mean, std)

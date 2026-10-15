"""FSatten: Frequency Spectrum attention inside a Variate (inverted) Transformer.

Independent implementation from Sections 3.1 and 3.2 of Wu, "Revisiting Attention
for Multivariate Time Series Forecasting" (arXiv 2407.13806). No official source code
was released (the official repository holds only training logs), so tensor layouts
not fixed by the paper are local choices recorded in the model card.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.revin import RevIN
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer
from tsflab.models._inputs import check_window

FFTNorm = Literal["backward", "ortho", "forward"]


def amplitude_spectrum(series: torch.Tensor, fft_norm: FFTNorm = "backward") -> torch.Tensor:
    """Eqs. (3)-(4): per-variate rFFT over time and its modulus.

    ``series`` is ``[batch, time, channels]``; the result is ``[batch, channels, F]``
    with ``F = time // 2 + 1`` real, non-negative amplitudes (phase is discarded).
    """
    spectrum = torch.fft.rfft(series.transpose(1, 2).contiguous(), dim=-1, norm=fft_norm)
    return spectrum.abs()


class MultiHeadSpectrumScaling(nn.Module):
    """Eq. (6): ``MSS(A) = A o W_h`` with one ``[C, F]`` scale matrix per head.

    A Hadamard product instead of a dense projection, so every frequency axis stays
    a frequency axis. Scales start at one (local choice: the paper gives no
    initialization), i.e. attention initially compares raw amplitude spectra.
    """

    def __init__(self, n_heads: int, channels: int, n_freq: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_heads, channels, n_freq))

    def forward(self, amplitude: torch.Tensor) -> torch.Tensor:
        # [B, C, F] -> [B, H, C, F]
        return amplitude.unsqueeze(1) * self.weight.unsqueeze(0)


class FrequencySpectrumAttention(nn.Module):
    """FSatten (Eq. 5): Q and K from MSS of the amplitude spectrum, V from the tokens.

    Follows the injected-attention API of ``transformer_encdec`` (``(q, k, v,
    attn_mask, tau, delta) -> (out, attn)``). The amplitude spectrum ``[B, C, F]``
    of the normalized input series arrives through ``delta``; the queries and keys
    tensors are ignored because Q and K never see the latent tokens.
    """

    def __init__(self, d_model: int, n_heads: int, channels: int, n_freq: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.channels = channels
        self.n_freq = n_freq
        self.query_scaling = MultiHeadSpectrumScaling(n_heads, channels, n_freq)
        self.key_scaling = MultiHeadSpectrumScaling(n_heads, channels, n_freq)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # dot-product scale: the per-head query/key width is F (1/sqrt(d_K)).
        self.scale = 1.0 / math.sqrt(n_freq)

    def attention_weights(self, amplitude: torch.Tensor) -> torch.Tensor:
        """``softmax(MSS_Q(A) MSS_K(A)^T / sqrt(F))`` as ``[B, H, C, C]`` (pre-dropout)."""
        check_window(amplitude, self.channels, self.n_freq, name="amplitude")
        queries = self.query_scaling(amplitude)
        keys = self.key_scaling(amplitude)
        scores = torch.einsum("bhcf,bhsf->bhcs", queries, keys) * self.scale
        return torch.softmax(scores, dim=-1)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        del queries, keys, attn_mask, tau
        if delta is None:
            raise ValueError("FrequencySpectrumAttention needs the amplitude spectrum as `delta`")
        batch, tokens, width = values.shape
        if tokens != self.channels:
            raise ValueError(f"expected {self.channels} variate tokens, got {tokens}")
        weights = self.attention_weights(delta)
        heads = self.value_projection(values).view(batch, tokens, self.n_heads, width // self.n_heads)
        mixed = torch.einsum("bhcs,bshd->bchd", self.dropout(weights), heads)
        return self.out_projection(mixed.reshape(batch, tokens, width)), weights


class Model(nn.Module):
    """Variate Transformer (iTransformer layout) with every attention replaced by FSatten."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
        fft_norm: FFTNorm = "backward",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.n_freq = seq_len // 2 + 1
        self.fft_norm = fft_norm
        self.normalizer = RevIN(enc_in, affine=False, enabled=use_norm)
        # value path: one whole lookback series per token (Emb in Eq. 5).
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FrequencySpectrumAttention(d_model, n_heads, enc_in, self.n_freq, dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.normalizer(x_enc, "norm")
        amplitude = amplitude_spectrum(normalized, self.fft_norm)
        tokens = self.embedding(normalized, None)
        tokens, _ = self.encoder(tokens, delta=amplitude)
        forecast = self.projection(tokens).transpose(1, 2)
        return self.normalizer(forecast, "denorm")

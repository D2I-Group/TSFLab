"""Minusformer: progressive residual learning with subtraction-only aggregation.

Independent implementation from Section 2.4 (Eqs. 5-9, Fig. 4) and Appendix H
(Algorithm 1) of Liang et al., "Minusformer: Improving Time Series Forecasting by
Progressively Learning Residuals" (arXiv 2402.02332), after reading the pinned
official code (``Anoise/Minusformer`` at ``78b3031c``, no license file) to resolve
omissions; nothing is copied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.marks import encoder_timef_marks
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention


class SeriesScaler:
    """Algorithm 1 lines 1 and 26: per-variate standardization over the lookback
    window (population variance plus ``1e-5`` under the root). The statistics keep
    their gradient, as in the official scaler."""

    def __init__(self, series: torch.Tensor, eps: float = 1e-5) -> None:
        # series: [B, N, L] variate tokens.
        self.mean = series.mean(-1, keepdim=True)
        self.std = torch.sqrt((series - self.mean).var(-1, keepdim=True, unbiased=False) + eps)

    def transform(self, series: torch.Tensor) -> torch.Tensor:
        return (series - self.mean) / self.std

    def invert(self, series: torch.Tensor) -> torch.Tensor:
        return series * self.std + self.mean


class MinusBlock(nn.Module):
    """One Minusformer block (Eqs. 5-9): a fork with a residual input stream and
    a gated output stream.

    ``R1 = X - Dropout(Attention(X))`` (Eq. 7, the attention term is dropped when
    ``attn`` is off), ``X2 = LayerNorm(R1)``, ``F = FeedForward(X2)``,
    ``R2 = X2 - F`` (Eq. 5b); the next input is ``LayerNorm(sigmoid(th1 R2) * th2 R2)``
    (Eq. 8) and the block output is ``sigmoid(th3 [A, F]) * th4 [A, F]`` (Eq. 9)
    projected to ``d_block``.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, d_block: int, dropout: float, attn: bool, gate: bool
    ) -> None:
        super().__init__()
        self.use_gate = gate
        self.attention = (
            AttentionLayer(
                FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False),
                d_model,
                n_heads,
            )
            if attn
            else None
        )
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.residual_gate = nn.Linear(d_model, d_model)
        self.residual_value = nn.Linear(d_model, d_model)
        width = 2 * d_model if attn else d_model
        self.output_gate = nn.Linear(width, d_block)
        self.output_value = nn.Linear(width, d_block)
        self.input_norm = nn.LayerNorm(d_model)
        self.residual_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Official choice: GELU with attention, ReLU for the attention-free block.
        self.activation = F.gelu if attn else F.relu

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ff_out(self.dropout(self.activation(self.ff_in(x)))))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` ``[B, tokens, d_model]`` -> (next input stream, block output ``[B, tokens, d_block]``)."""
        attended = None
        if self.attention is not None:
            attended, _ = self.attention(x, x, x, None)
            x = x - self.dropout(attended)  # Eq. (7b)
        x = self.input_norm(x)
        transformed = self.feed_forward(x)
        residual = x - transformed  # Eq. (5b)
        gate = torch.sigmoid(self.residual_gate(residual)) if self.use_gate else 1.0
        next_input = self.residual_norm(gate * self.residual_value(residual))  # Eq. (8)
        branch = torch.cat((attended, transformed), dim=-1) if attended is not None else transformed
        out_gate = torch.sigmoid(self.output_gate(branch)) if self.use_gate else 1.0
        return next_input, out_gate * self.output_value(branch)  # Eq. (9)


def subtract_outputs(block_outputs: list[torch.Tensor]) -> torch.Tensor:
    """Output stream (Algorithm 1 line 21): ``O_{l+1} = O_hat_{l+1} - O_l`` from
    ``O_0 = 0``, i.e. Eq. (2) with alternating signs ending positive on the last block."""
    total: torch.Tensor | float = 0.0
    for output in block_outputs:
        total = output - total
    return total  # type: ignore[return-value]


class Model(nn.Module):
    """Minusformer forecaster for ``[B, seq_len, enc_in]`` histories (variate tokens)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        d_block: int | None = None,
        dropout: float = 0.1,
        attn: bool = True,
        gate: bool = True,
        use_marks: bool = True,
        freq: str = "h",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("Minusformer sizes must be positive")
        d_block = pred_len if d_block is None else d_block
        if d_block < 1:
            raise ValueError("d_block must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_marks = use_marks
        self.freq = freq
        # Algorithm 1 line 3: one linear map of each whole (scaled) series; no dropout.
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=0.0)
        self.blocks = nn.ModuleList(
            MinusBlock(d_model, n_heads, d_ff, d_block, dropout, attn, gate) for _ in range(e_layers)
        )
        # Algorithm 1 lines 24-25: align the block length with the horizon.
        self.align = nn.Linear(d_block, pred_len) if d_block != pred_len else nn.Identity()

    def calendar_tokens(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        return encoder_timef_marks(x_mark_enc, seq_len=self.seq_len, freq=self.freq, enabled=self.use_marks)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the block stack and return the subtracted output stream."""
        outputs = []
        for block in self.blocks:
            tokens, out = block(tokens)
            outputs.append(out)
        return subtract_outputs(outputs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in})")
        series = x_enc.transpose(1, 2)  # [B, N, L]
        scaler = SeriesScaler(series)
        tokens = self.embedding(scaler.transform(series).transpose(1, 2), self.calendar_tokens(x_mark_enc))
        output = self.align(self.encode(tokens))[:, : self.enc_in]  # drop calendar tokens
        return scaler.invert(output).transpose(1, 2)

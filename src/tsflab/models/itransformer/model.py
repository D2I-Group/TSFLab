"""Clean-room iTransformer with whole-variate tokens and native attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.instance_norm import destandardize, standardize


class InvertedEmbedding(nn.Module):
    """Paper equation (1): one complete lookback series becomes one token."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.projection = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _scaled_marks(marks: torch.Tensor) -> torch.Tensor:
        if marks.ndim != 3 or marks.shape[-1] != 6:
            raise ValueError("calendar marks must have shape (batch, time, 6)")
        scales = marks.new_tensor((2100.0, 12.0, 31.0, 6.0, 23.0, 59.0))
        return marks / scales - 0.5

    def forward(self, values: torch.Tensor, marks: torch.Tensor | None) -> torch.Tensor:
        series_tokens = values.transpose(1, 2)
        if marks is not None:
            series_tokens = torch.cat(
                (series_tokens, self._scaled_marks(marks).transpose(1, 2)), dim=1
            )
        return self.dropout(self.projection(series_tokens))


class InvertedEncoderLayer(nn.Module):
    """Native Transformer block operating across variate tokens."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        nonlinear: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nonlinear,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, return_attention: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        tokens = self.attention_norm(tokens + self.dropout(attended))
        tokens = self.feed_forward_norm(tokens + self.dropout(self.feed_forward(tokens)))
        return tokens, weights if return_attention else None


class Model(nn.Module):
    """Encoder-only iTransformer forecaster for a fixed catalog channel contract."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int, n_heads: int, e_layers: int, d_ff: int, dropout: float, activation: str, output_attention: bool, use_norm: bool) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, n_heads, e_layers, d_ff) < 1:
            raise ValueError("lengths, channels, widths, heads, and layers must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.embedding = InvertedEmbedding(seq_len, d_model, dropout)
        self.encoder = nn.ModuleList(
            [
                InvertedEncoderLayer(
                    d_model, n_heads, d_ff, dropout, activation
                )
                for _ in range(e_layers)
            ]
        )
        self.projection = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.channels})")
        if x_mark_enc is not None and x_mark_enc.shape != (
            x_enc.shape[0],
            self.seq_len,
            6,
        ):
            raise ValueError("encoder marks must align with x_enc and contain six columns")
        if self.use_norm:
            normalized, mean, stdev = standardize(x_enc)
        else:
            normalized = x_enc
            mean = x_enc.new_zeros(x_enc.shape[0], 1, self.channels)
            stdev = x_enc.new_ones(x_enc.shape[0], 1, self.channels)

        tokens = self.embedding(normalized, x_mark_enc)
        attention_maps = []
        for layer in self.encoder:
            tokens, attention = layer(tokens, self.output_attention)
            if attention is not None:
                attention_maps.append(attention)
        variable_tokens = tokens[:, : self.channels]
        forecast = self.projection(variable_tokens).transpose(1, 2)
        forecast = destandardize(forecast, mean, stdev)
        if self.output_attention:
            return forecast, attention_maps
        return forecast

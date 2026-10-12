"""Local Ister (CD variant) written from the paper and a review of the pinned code.

Ister decomposes the instance-normalized window into seasonal and trend parts
(Eq. 2), embeds every variate's whole seasonal and trend series as one token,
runs the seasonal tokens through a Transformer encoder whose attention core is
Dot-attention (Eq. 1, linear in the number of tokens), refines the trend tokens
with a residual MLP, and sums two linear projections to the horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import adapt_tslib_marks
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer
from tsflab.models._components.series_decomposition import SeriesDecomposition
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer
from tsflab.models._inputs import check_window


class DotAttention(nn.Module):
    """Eq. (1): ``Dot(Q, K, V) = (sum_i Softmax(Q)_i * K_i)^T 1_N * V``.

    ``Softmax(Q)`` normalizes every feature over the token (variate) axis, the
    weighted keys are summed over tokens into one global ``[D]`` vector, and that
    vector gates every value token element-wise, so the cost is ``O(N D)``.
    With ``gelu_scores`` (official code) a GELU is applied to the per-token
    products before the token sum; the paper's Eq. (1) has no nonlinearity there.
    Accepts and ignores the mask/tau/delta arguments of the attention-core API.
    """

    def __init__(self, attention_dropout: float = 0.1, gelu_scores: bool = True) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.gelu_scores = gelu_scores

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        del attn_mask, tau, delta
        batch, length = queries.shape[:2]
        if keys.shape[1] != length or values.shape[1] != length:
            raise ValueError("Dot-attention is self-attention: queries, keys and values need equal token counts")
        q = queries.reshape(batch, length, -1)
        k = keys.reshape(batch, length, -1)
        v = values.reshape(batch, length, -1)
        weighted = torch.softmax(q, dim=1) * k
        if self.gelu_scores:
            weighted = F.gelu(weighted)
        global_score = weighted.sum(dim=1, keepdim=True)  # [B, 1, H*E]
        out = self.dropout(v * global_score)
        return out.view_as(values), None


class TrendMLP(nn.Module):
    """Residual two-layer GELU MLP with a linear shortcut and LayerNorm.

    ``LN(W2 drop(GELU(W1 h)) + W3 h)`` on each trend token; the model adds it to
    its input (``h + TrendMLP(h)``), the paper's ``MLP(X_t)`` of Algorithm 1.
    """

    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.shortcut = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.fc2(F.gelu(self.fc1(tokens))))
        return self.norm(out + self.shortcut(tokens))


class InvertedSeriesEmbedding(nn.Module):
    """Whole-series variate tokens: ``Linear(T -> D)`` over each channel and,
    when marks are given, over each calendar feature (iTransformer layout)."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, marks: torch.Tensor | None) -> torch.Tensor:
        tokens = values.transpose(1, 2)
        if marks is not None:
            tokens = torch.cat((tokens, marks.transpose(1, 2).to(tokens.dtype)), dim=1)
        return self.dropout(self.projection(tokens))


class Model(nn.Module):
    """Inverted seasonal-trend decomposition Transformer with Dot-attention."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_ff: int = 2048,
        layers: int = 2,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_marks: bool = True,
        freq: str = "h",
        recon_len: int = 48,
        gelu_scores: bool = True,
    ) -> None:
        super().__init__()
        if recon_len > seq_len:
            raise ValueError("recon_len must not exceed seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_marks = use_marks
        self.freq = freq
        self.recon_len = recon_len
        self.out_len = recon_len + pred_len

        self.normalizer = RevIN(enc_in, eps=1e-5, affine=False)
        self.decomposition = SeriesDecomposition(moving_avg)
        self.seasonal_embedding = InvertedSeriesEmbedding(seq_len, d_model, dropout)
        self.trend_embedding = InvertedSeriesEmbedding(seq_len, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(DotAttention(dropout, gelu_scores), d_model, 1),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.trend_mlp = TrendMLP(d_model, d_model // 2, dropout)
        self.seasonal_head = nn.Linear(d_model, self.out_len)
        self.trend_head = nn.Linear(d_model, self.out_len)
        # Official initialization: both heads start as uniform token averages (1 / D).
        for head in (self.seasonal_head, self.trend_head):
            nn.init.constant_(head.weight, 1.0 / d_model)

    def _marks(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_marks or x_mark_enc is None:
            return None
        return adapt_tslib_marks(x_mark_enc, embed_type="timeF", freq=self.freq)

    def full_forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``[B, recon_len + pred_len, C]``: the reconstruction of the last
        ``recon_len`` history steps followed by the horizon forecast."""
        check_window(x_enc, self.seq_len, self.enc_in)
        marks = self._marks(x_mark_enc)
        normalized = self.normalizer(x_enc, "norm")
        seasonal, trend = self.decomposition(normalized)
        seasonal_tokens = self.seasonal_embedding(seasonal, marks)
        trend_tokens = self.trend_embedding(trend, marks)
        seasonal_tokens, _ = self.encoder(seasonal_tokens, attn_mask=None)
        trend_tokens = trend_tokens + self.trend_mlp(trend_tokens)
        seasonal_out = self.seasonal_head(seasonal_tokens).transpose(1, 2)[:, :, : self.enc_in]
        trend_out = self.trend_head(trend_tokens).transpose(1, 2)[:, :, : self.enc_in]
        return self.normalizer(seasonal_out + trend_out, "denorm")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        return self.full_forecast(x_enc, x_mark_enc)[:, -self.pred_len :, :]


__all__ = ["DotAttention", "InvertedSeriesEmbedding", "Model", "TrendMLP"]

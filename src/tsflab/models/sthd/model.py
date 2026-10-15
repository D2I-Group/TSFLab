"""Paper-driven local implementation of STHD.

STHD (Scalable Transformer for High-Dimensional MTS forecasting) forecasts each
target series from itself plus its ``K`` most Pearson-correlated series on the
training split (Relation Matrix Sparsity, Sec. 4.1). Every (target, related)
group is patched, embedded with a time and a channel sinusoidal position code,
and encoded jointly by a vanilla Transformer whose self-attention spans all
``(1 + K) * P`` patch tokens (2-D Transformer, Sec. 4.3); the target's patch
representations are flattened and projected to the horizon.

The relation is a property of the training data, not a learned weight: it is
estimated once from the scaled training series (:meth:`Model.fit_relations`)
and stored in the ``related`` buffer. Before fitting, ``related`` holds the
cyclically next ``K`` channels so the model is always runnable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.embed import PatchEmbedding
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer
from tsflab.models._inputs import check_window


def channel_position_table(length: int, d_model: int, base: float = 5000.0) -> torch.Tensor:
    """Sinusoidal code over the channel axis (index 0 is the target series)."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(base) / d_model))
    table = torch.zeros(length, d_model)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
    return table


@torch.no_grad()
def top_k_related(series: torch.Tensor, k: int) -> torch.Tensor:
    """``[T, N] -> [N, k]``: each channel's ``k`` most correlated other channels.

    Pearson correlation of the z-normalized series; undefined correlations
    (constant channels) count as 0; the channel itself is excluded; ties keep
    the lower index first.
    """
    if series.ndim != 2 or series.shape[0] < 2:
        raise ValueError(f"expected a [time, channels] series with time >= 2, got {tuple(series.shape)}")
    channels = series.shape[1]
    if not 0 <= k < channels:
        raise ValueError(f"k must lie in [0, {channels - 1}], got {k}")
    values = series.double()
    centered = values - values.mean(dim=0, keepdim=True)
    std = centered.pow(2).mean(dim=0).sqrt()
    z = centered / std
    corr = (z.transpose(0, 1) @ z) / values.shape[0]
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr.fill_diagonal_(-math.inf)
    order = torch.sort(corr, dim=1, descending=True, stable=True).indices
    return order[:, :k].long()


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        k: int = 13,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 384,
        dropout: float = 0.2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.k = min(k, enc_in - 1)
        self.revin = RevIN(enc_in, eps=1e-5, affine=False)
        # Eq. (1)-(2): end-replicated padding of one stride, P = floor((L - l) / s) + 2 patches,
        # a bias-free linear patch projection plus the time sinusoidal code (dropout applied below).
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, stride, dropout=0.0)
        self.patch_num = (seq_len - patch_len) // stride + 2
        self.register_buffer("channel_position", channel_position_table(self.k + 1, d_model), persistent=False)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.head = FlattenForecastHead(False, enc_in, d_model * self.patch_num, pred_len, head_dropout=dropout)
        channels = torch.arange(enc_in)
        cyclic = (channels.unsqueeze(1) + torch.arange(1, self.k + 1).unsqueeze(0)) % enc_in
        self.register_buffer("related", cyclic.long())
        self.register_buffer("relations_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_relations(self, series: torch.Tensor) -> None:
        """Relation Matrix Sparsity: keep each channel's top-``K`` correlated series."""
        if series.ndim != 2 or series.shape[1] != self.enc_in:
            raise ValueError(f"expected a [time, {self.enc_in}] series, got {tuple(series.shape)}")
        self.related.copy_(top_k_related(series, self.k).to(self.related.device))
        self.relations_fitted.fill_(True)

    def group_index(self) -> torch.Tensor:
        """``[N, 1 + K]``: the target channel followed by its related channels."""
        targets = torch.arange(self.enc_in, device=self.related.device).unsqueeze(1)
        return torch.cat((targets, self.related), dim=1)

    def embed_groups(self, groups: torch.Tensor) -> torch.Tensor:
        """``[G, 1 + K, L] -> [G, 1 + K, P, D]`` with time and channel position codes (Eqs. 2-3)."""
        tokens, members = self.patch_embedding(groups)
        tokens = tokens.reshape(groups.shape[0], members, self.patch_num, -1)
        tokens = tokens + self.channel_position[:, None, :]
        return self.embedding_dropout(tokens)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        # Per-series instance normalization (identical whether applied before or after grouping).
        normalized = self.revin(x_enc, "norm")
        groups = normalized[:, :, self.group_index()]  # [B, L, N, 1 + K]
        groups = groups.permute(0, 2, 3, 1).reshape(batch * self.enc_in, self.k + 1, self.seq_len)
        tokens = self.embed_groups(groups)
        width = tokens.shape[-1]
        # One attention over every patch of the target and its related series.
        encoded, _ = self.encoder(tokens.reshape(tokens.shape[0], -1, width))
        target = encoded.reshape(batch, self.enc_in, self.k + 1, self.patch_num, width)[:, :, 0]
        forecast = self.head(target.transpose(-1, -2))  # [B, N, H]
        return self.revin(forecast.transpose(1, 2), "denorm")

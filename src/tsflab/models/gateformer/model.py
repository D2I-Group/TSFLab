"""Clean-room local implementation of Gateformer.

Gateformer builds two complementary per-variate embeddings and lets a
learned gate combine them (paper Section 3, Figure 2):

* a *global* embedding: one token per variate, produced by projecting the
  whole (normalized) lookback window (the "inverted" embedding also used by
  iTransformer);
* a *temporal-dependency* embedding: each variate is patched, a shared
  Transformer encoder attends across patches (cross-time attention), and a
  flatten-and-project head reduces the encoded patches back to one token per
  variate.

A first gate fuses these into one variate embedding per channel. A second,
variate-wise Transformer encoder then attends across variates
(cross-variate attention), and a second gate blends its output with its
input before a final linear head projects each variate token to the forecast
horizon.
"""

from __future__ import annotations

import torch.nn as nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.gated_fusion import GatedFusion
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.positional_encoding import positional_encoding


def _encoder(d_model: int, n_heads: int, d_ff: int, dropout: float, e_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=e_layers)


class Model(nn.Module):
    """Two-branch, gate-fused temporal/variate attention forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride

        self.patch_num = (seq_len + stride - patch_len) // stride + 1
        self.patch_padding = nn.ReplicationPad1d((0, stride))
        self.patch_projection = nn.Linear(patch_len, d_model, bias=False)
        self.patch_position = positional_encoding("sincos", False, self.patch_num, d_model)
        self.patch_dropout = nn.Dropout(dropout)

        self.global_projection = nn.Linear(seq_len, d_model)
        self.global_dropout = nn.Dropout(dropout)

        self.temporal_encoder = _encoder(d_model, n_heads, d_ff, dropout, e_layers)
        self.temporal_head = FlattenForecastHead(
            individual=False, n_vars=enc_in, nf=d_model * self.patch_num, target_window=d_model, head_dropout=dropout
        )
        self.gate_global_temporal = GatedFusion(d_model)

        self.variate_encoder = _encoder(d_model, n_heads, d_ff, dropout, e_layers)
        self.gate_variate = GatedFusion(d_model)

        self.projection = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in})")

        normalized, mean, stdev = standardize(x_enc)

        # Global (inverted) embedding: one token per variate over the whole window.
        global_embedding = self.global_dropout(self.global_projection(normalized.transpose(1, 2)))

        # Temporal-dependency embedding: patch each variate, attend across patches.
        channels_first = normalized.transpose(1, 2)
        padded = self.patch_padding(channels_first)
        patches = padded.unfold(-1, self.patch_len, self.stride)
        batch, n_vars, patch_num, _ = patches.shape
        tokens = patches.reshape(batch * n_vars, patch_num, self.patch_len)
        tokens = self.patch_dropout(self.patch_projection(tokens) + self.patch_position)
        encoded = self.temporal_encoder(tokens)
        encoded = encoded.reshape(batch, n_vars, patch_num, -1).permute(0, 1, 3, 2)
        temporal_embedding = self.temporal_head(encoded)

        fused = self.gate_global_temporal(global_embedding, temporal_embedding)

        cross_variate = self.variate_encoder(fused)
        fused = self.gate_variate(fused, cross_variate)

        forecast = self.projection(fused).transpose(1, 2)
        return destandardize(forecast, mean, stdev)

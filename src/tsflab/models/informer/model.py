"""Paper-driven local Informer forecaster.

Informer combines query-sparse attention, encoder distilling, and a
generative one-shot decoder. This module assembles those paper-level ideas
from TSFLab's generic attention, embedding, and transformer components.
"""

from __future__ import annotations

import torch.nn as nn

from tsflab.models._components.embed import DataEmbedding
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention, ProbAttention
from tsflab.models._components.transformer_encdec import ConvLayer, Decoder, DecoderLayer, Encoder, EncoderLayer


class Model(nn.Module):
    """ProbSparse encoder-decoder for long-sequence forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        label_len: int,
        features: str,
        enc_in: int,
        dec_in: int | None = None,
        c_out: int | None = None,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        factor: int = 3,
        activation: str = "gelu",
        distil: bool = True,
        embed: str = "timeF",
        freq: str = "h",
    ) -> None:
        super().__init__()
        del seq_len, label_len, features
        dec_in = enc_in if dec_in is None else dec_in
        c_out = enc_in if c_out is None else c_out
        self.pred_len = pred_len
        self.encoder_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.decoder_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        def sparse(*, causal: bool) -> AttentionLayer:
            return AttentionLayer(
                ProbAttention(mask_flag=causal, factor=factor, attention_dropout=dropout),
                d_model,
                n_heads,
            )

        def full() -> AttentionLayer:
            return AttentionLayer(
                FullAttention(mask_flag=False, attention_dropout=dropout),
                d_model,
                n_heads,
            )

        layers = [EncoderLayer(sparse(causal=False), d_model, d_ff, dropout, activation) for _ in range(e_layers)]
        distillers = [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil and e_layers > 1 else None
        self.encoder = Encoder(layers, conv_layers=distillers, norm_layer=nn.LayerNorm(d_model))
        self.decoder = Decoder(
            [DecoderLayer(sparse(causal=True), full(), d_model, d_ff, dropout, activation) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        memory, _ = self.encoder(self.encoder_embedding(x_enc, x_mark_enc))
        decoded = self.decoder(self.decoder_embedding(x_dec, x_mark_dec), memory)
        return decoded[:, -self.pred_len :, :]

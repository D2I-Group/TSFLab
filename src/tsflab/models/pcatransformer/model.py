"""PCATransformer: PCA-reduced covariates feeding a vanilla Transformer (AI4TS@AAAI 2024).

Xu et al. forecast one target from ``M`` covariates (the ``MS`` setting). Before the
forecaster sees the data, the standardized covariates are replaced by their first
``k`` principal components while the target channel is kept unchanged, so a
``[T, M + 1]`` series becomes ``[T, k + 1]``. The reduced series is forecast by the
vanilla encoder-decoder Transformer, whose decoder projects to the single target.

The principal axes are a property of the training data, not learned weights: they are
estimated once from the scaled training series (:meth:`Model.fit_pca`) and stored as
buffers. Until then the projection keeps the first ``k`` covariates unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.embed import DataEmbedding
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Decoder, DecoderLayer, Encoder, EncoderLayer
from tsflab.models._inputs import check_window


def principal_axes(covariates: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean ``[M]`` and the top ``n_components`` principal axes ``[k, M]`` of ``[T, M]`` rows.

    PCA by singular value decomposition of the centred rows; axes are ordered by
    explained variance. Each axis is oriented so that its largest-magnitude loading is
    positive (PCA axes are only defined up to sign).
    """
    if covariates.ndim != 2:
        raise ValueError("covariates must be a [time, variables] matrix")
    rows, variables = covariates.shape
    if not 1 <= n_components <= min(rows, variables):
        raise ValueError(
            f"n_components={n_components} must be in [1, min(time={rows}, variables={variables})]"
        )
    data = covariates.double()
    mean = data.mean(dim=0)
    _, _, vh = torch.linalg.svd(data - mean, full_matrices=False)
    axes = vh[:n_components]
    pivot = axes.abs().argmax(dim=1, keepdim=True)
    signs = torch.sign(axes.gather(1, pivot))
    signs[signs == 0] = 1.0
    return mean.to(covariates.dtype), (axes * signs).to(covariates.dtype)


class Model(nn.Module):
    """PCA covariate reduction + full-attention encoder-decoder for one target channel."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        label_len: int,
        enc_in: int,
        n_components: int = 2,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        embed: str = "timeF",
        freq: str = "h",
    ) -> None:
        super().__init__()
        if label_len > seq_len:
            raise ValueError("label_len must not exceed seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        self.n_components = n_components
        reduced = n_components + 1
        # Placeholder projection (first k covariates, zero mean) until fit_pca runs.
        self.register_buffer("pca_mean", torch.zeros(enc_in - 1))
        self.register_buffer("pca_axes", torch.eye(enc_in - 1)[:n_components].clone())
        self.register_buffer("pca_fitted", torch.tensor(False))

        self.encoder_embedding = DataEmbedding(reduced, d_model, embed, freq, dropout)
        self.decoder_embedding = DataEmbedding(reduced, d_model, embed, freq, dropout)

        def attention(*, causal: bool) -> AttentionLayer:
            return AttentionLayer(
                FullAttention(mask_flag=causal, attention_dropout=dropout), d_model, n_heads
            )

        self.encoder = Encoder(
            [EncoderLayer(attention(causal=False), d_model, d_ff, dropout, activation)
             for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )
        # c_out = 1: the decoder forecasts only the target channel (official MS runs).
        self.decoder = Decoder(
            [DecoderLayer(attention(causal=True), attention(causal=False), d_model, d_ff,
                          dropout, activation)
             for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, 1),
        )

    @torch.no_grad()
    def fit_pca(self, series: torch.Tensor) -> None:
        """Estimate the covariate principal axes from a scaled ``[T, enc_in]`` series.

        The last column is the target and is excluded; the remaining ``enc_in - 1``
        covariates define the PCA.
        """
        if series.ndim != 2 or series.shape[1] != self.enc_in:
            raise ValueError(f"expected a [time, {self.enc_in}] series, got {tuple(series.shape)}")
        mean, axes = principal_axes(series[:, :-1].to(self.pca_mean), self.n_components)
        self.pca_mean.copy_(mean)
        self.pca_axes.copy_(axes)
        self.pca_fitted.fill_(True)

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        """``[B, L, enc_in] -> [B, L, k + 1]``: principal-component scores, then the target."""
        scores = (values[..., :-1] - self.pca_mean) @ self.pca_axes.transpose(0, 1)
        return torch.cat((scores, values[..., -1:]), dim=-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        encoder_input = self.reduce(x_enc)
        # Decoder input in the reduced space: observed label window, then a zero horizon
        # (the official loader applies PCA before the zero horizon is appended).
        if x_dec is not None:
            label_len = x_dec.shape[1] - self.pred_len
            if label_len < 0:
                raise ValueError("x_dec must cover the forecast horizon")
            label = self.reduce(x_dec[:, :label_len])
        else:
            label = encoder_input[:, self.seq_len - self.label_len :]
        horizon = encoder_input.new_zeros(x_enc.shape[0], self.pred_len, self.n_components + 1)
        decoder_input = torch.cat((label, horizon), dim=1)
        memory, _ = self.encoder(self.encoder_embedding(encoder_input, x_mark_enc))
        decoded = self.decoder(self.decoder_embedding(decoder_input, x_mark_dec), memory)
        return decoded[:, -self.pred_len :, :]

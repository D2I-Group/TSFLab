"""Independent HN-MVTS implementation from the paper equations.

The defining operation is a partial hypernetwork: a learnable channel
embedding is mapped to the forecasting backbone's final-layer weights. This
module uses a small channel-independent temporal encoder as that backbone; it
does not copy the reference repository or reproduce its training pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 embedding_dim: int = 8, hyper_hidden: int = 32,
                 use_revin: bool = True) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.d_model, self.use_revin = d_model, use_revin
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.temporal_encoder = nn.Sequential(nn.Linear(seq_len, d_model), nn.GELU())
        self.channel_embedding = nn.Parameter(torch.empty(enc_in, embedding_dim))
        self.hypernetwork = nn.Sequential(
            nn.Linear(embedding_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, pred_len * d_model + pred_len),
        )
        nn.init.normal_(self.channel_embedding, std=0.02)

    def generated_projection(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return paper Eq. (3)'s channel-specific final-layer parameters."""
        generated = self.hypernetwork(self.channel_embedding)
        split = self.pred_len * self.d_model
        weights = generated[:, :split].reshape(self.enc_in, self.pred_len, self.d_model)
        bias = generated[:, split:].reshape(self.enc_in, self.pred_len)
        return weights, bias

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        normalized = self.revin(x_enc, "norm") if self.use_revin else x_enc
        hidden = self.temporal_encoder(normalized.transpose(1, 2))
        weights, bias = self.generated_projection()
        forecast = torch.einsum("bcd,chd->bch", hidden, weights) + bias.unsqueeze(0)
        forecast = forecast.transpose(1, 2)
        return self.revin(forecast, "denorm") if self.use_revin else forecast

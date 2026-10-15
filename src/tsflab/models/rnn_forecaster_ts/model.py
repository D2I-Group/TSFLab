"""Independent Elman-RNN forecasting baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Encode a history with a tanh RNN and decode its final state directly."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in, enabled=use_revin)
        self.encoder = nn.RNN(
            enc_in,
            d_model,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_model, pred_len * enc_in)
        self.aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        _, final_state = self.encoder(normalized)
        forecast = self.head(final_state[-1]).reshape(-1, self.pred_len, self.enc_in)
        forecast = self.revin(forecast, "denorm")
        self.aux_loss = forecast.new_zeros(())
        return forecast

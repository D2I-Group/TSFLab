"""Independent channel-wise MLP forecasting baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Map each channel's complete lag window directly to its forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in, enabled=use_revin)
        layers: list[nn.Module] = [nn.Linear(seq_len, d_model), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)))
        layers.append(nn.Linear(d_model, pred_len))
        self.network = nn.Sequential(*layers)
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
        forecast = self.network(normalized.transpose(1, 2)).transpose(1, 2)
        forecast = self.revin(forecast, "denorm")
        self.aux_loss = forecast.new_zeros(())
        return forecast

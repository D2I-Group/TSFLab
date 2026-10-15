"""Clean-room per-series LSTM forecasting baseline."""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window


def _fit_features(values: torch.Tensor, width: int) -> torch.Tensor:
    if values.shape[-1] >= width:
        return values[..., :width]
    padding = values.new_zeros((*values.shape[:-1], width - values.shape[-1]))
    return torch.cat((values, padding), dim=-1)


class Model(nn.Module):
    """Encode every node with shared LSTM gates and a direct horizon head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        init_dim: int = 32,
        hid_dim: int = 64,
        end_dim: int = 128,
        layer: int = 2,
        dropout: float = 0.1,
        cov_dim: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.input_dim = 1 + cov_dim
        self.input_projection = nn.Linear(self.input_dim, init_dim)
        self.recurrent = nn.LSTM(
            init_dim,
            hid_dim,
            num_layers=layer,
            batch_first=True,
            dropout=dropout if layer > 1 else 0.0,
        )
        self.forecast = nn.Sequential(nn.Linear(hid_dim, end_dim), nn.GELU(), nn.Linear(end_dim, pred_len))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        features = _fit_features(to_spatiotemporal(x_enc, x_mark_enc), self.input_dim)
        batch = features.shape[0]
        node_sequences = features.permute(0, 2, 1, 3).reshape(batch * self.enc_in, self.seq_len, self.input_dim)
        encoded, _ = self.recurrent(self.input_projection(node_sequences))
        prediction = self.forecast(encoded[:, -1])
        return prediction.view(batch, self.enc_in, self.pred_len).transpose(1, 2)

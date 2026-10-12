"""Historical-last persistence baseline."""

from __future__ import annotations

from torch import nn


class Model(nn.Module):
    """Repeat each series' final observation across the forecast horizon."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("sequence length, horizon, and channel count must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]"
            )
        return x_enc[:, -1:, :].expand(-1, self.pred_len, -1).clone()

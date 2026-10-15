"""Historical-last persistence baseline."""

from __future__ import annotations

from torch import nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Repeat each series' final observation across the forecast horizon."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int) -> None:
        super().__init__()
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
        check_window(x_enc, self.seq_len, self.enc_in)
        return x_enc[:, -1:, :].expand(-1, self.pred_len, -1).clone()

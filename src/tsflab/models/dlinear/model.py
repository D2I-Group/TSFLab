"""Paper-driven local implementation of decomposition-linear forecasting."""

from __future__ import annotations

from torch import nn

from tsflab.models._components.dlinear import DLinearBackbone


class Model(nn.Module):
    """Sum linear forecasts of moving-average trend and seasonal remainder."""

    def __init__(self, c_in: int, seq_len: int, pred_len: int,
                 kernel_size: int = 25, individual: bool = False) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("channels and sequence lengths must be positive")
        self.c_in = c_in
        self.seq_len = seq_len
        self.backbone = DLinearBackbone(
            c_in, seq_len, pred_len, kernel_size=kernel_size, individual=individual
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.c_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        return self.backbone(x_enc)

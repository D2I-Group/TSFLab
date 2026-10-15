"""Paper-driven local implementation of decomposition-linear forecasting."""

from __future__ import annotations

from torch import nn

from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Sum linear forecasts of moving-average trend and seasonal remainder."""

    def __init__(self, c_in: int, seq_len: int, pred_len: int, kernel_size: int = 25, individual: bool = False) -> None:
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.backbone = DLinearBackbone(c_in, seq_len, pred_len, kernel_size=kernel_size, individual=individual)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.c_in)
        return self.backbone(x_enc)

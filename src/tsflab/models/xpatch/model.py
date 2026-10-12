"""Independent paper-based rewrite of xPatch."""

from __future__ import annotations

import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window
from tsflab.models.xpatch.layers import DualStreamForecaster, ExponentialDecomposition


class Model(nn.Module):
    """EMA-decomposed, channel-independent dual-stream forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str = "end",
        ma_type: str = "ema",
        alpha: float = 0.3,
        beta: float = 0.3,
        revin: bool = True,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, affine=revin, enabled=revin)
        self.decomposition = ExponentialDecomposition(alpha, beta, ma_type)
        self.forecaster = DualStreamForecaster(
            seq_len,
            pred_len,
            patch_len,
            stride,
            padding_patch,
            hidden_dim,
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        seasonal, trend = self.decomposition(normalized)
        batch = x_enc.shape[0]
        seasonal_ci = seasonal.transpose(1, 2).reshape(
            batch * self.enc_in, self.seq_len
        )
        trend_ci = trend.transpose(1, 2).reshape(batch * self.enc_in, self.seq_len)
        forecast = self.forecaster(seasonal_ci, trend_ci)
        forecast = forecast.reshape(batch, self.enc_in, self.pred_len).transpose(1, 2)
        return self.revin(forecast, "denorm")

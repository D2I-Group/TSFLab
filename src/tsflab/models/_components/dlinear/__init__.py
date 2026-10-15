"""Series decomposition and linear forecasting backbone used by DLinear methods."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.series_decomposition import SeriesDecomposition


class DLinearBackbone(nn.Module):
    """DLinear sequence-to-sequence forecasting backbone.

    Parameters
    ----------
    c_in : int
        Number of input channels.
    seq_len : int
        Input sequence length.
    pred_len : int
        Prediction horizon length.
    kernel_size : int, optional
        Kernel size for the moving average decomposition.
    individual : bool, optional
        Whether to use per-channel linear layers.
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = SeriesDecomposition(kernel_size)
        self.individual = individual
        self.channels = c_in

        self.seasonal_projection = ChannelWiseLinear(seq_len, pred_len, c_in, individual)
        self.trend_projection = ChannelWiseLinear(seq_len, pred_len, c_in, individual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a prediction sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, pred_len, C).
        """
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.seasonal_projection(seasonal_init)
        trend_output = self.trend_projection(trend_init)

        out = seasonal_output + trend_output
        return out.permute(0, 2, 1)

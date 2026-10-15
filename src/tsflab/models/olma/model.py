"""OLMA (arXiv:2505.11567): one frequency-domain loss for forecasting.

OLMA changes the supervision, not the architecture. The loss compares forecast
and label after (i) a DFT across channels at every time step (Eq. 11-12), (ii) a
DFT along time for every channel and (iii) a single-level Haar DWT along time for
every channel (Eq. 13-14), each with an L1 error, and combines them linearly
(Eq. 15). It replaces the time-domain loss during training. This entry carries it
on the DLinear backbone (the ``dlinear`` component), one of the paper's backbones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._components.haar_dwt1d import HaarDWT1D
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """DLinear forecaster trained with the OLMA loss."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_size: int = 25,
        individual: bool = False,
        loss_weight: float = 1.0,
        weight_channel: float = 0.34,
        weight_temporal: float = 0.33,
        weight_wavelet: float = 0.33,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.loss_weight = float(loss_weight)
        self.weight_channel = float(weight_channel)
        self.weight_temporal = float(weight_temporal)
        self.weight_wavelet = float(weight_wavelet)
        self.backbone = DLinearBackbone(enc_in, seq_len, pred_len, kernel_size, individual)
        self.wavelet = HaarDWT1D()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        return self.backbone(x_enc)

    @staticmethod
    def channel_loss(forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Eq. (11)-(12): mean |rDFT over channels| error per time step."""
        return (torch.fft.rfft(forecast, dim=-1) - torch.fft.rfft(target, dim=-1)).abs().mean()

    @staticmethod
    def temporal_fourier_loss(forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Eq. (13), first term: mean |rDFT over time| error per channel."""
        return (torch.fft.rfft(forecast, dim=1) - torch.fft.rfft(target, dim=1)).abs().mean()

    def temporal_wavelet_loss(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Eq. (13)-(14), second term: mean |Haar (cA, cD)| error per channel."""
        approx_f, detail_f = self.wavelet(forecast.transpose(1, 2))
        approx_t, detail_t = self.wavelet(target.transpose(1, 2))
        coefficients_f = torch.cat((approx_f, detail_f), dim=-1)
        coefficients_t = torch.cat((approx_t, detail_t), dim=-1)
        return (coefficients_f - coefficients_t).abs().mean()

    def olma_loss(self, forecast: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Eq. (15): ``w * (alpha L_c + beta L_tF + gamma L_tW)`` on ``[B, L, C]`` tensors."""
        if forecast.shape != target.shape or forecast.ndim != 3:
            raise ValueError("forecast and target must share a [batch, time, channels] shape")
        terms = {
            "channel": self.channel_loss(forecast, target),
            "temporal_fourier": self.temporal_fourier_loss(forecast, target),
            "temporal_wavelet": self.temporal_wavelet_loss(forecast, target),
        }
        total = (
            self.weight_channel * terms["channel"]
            + self.weight_temporal * terms["temporal_fourier"]
            + self.weight_wavelet * terms["temporal_wavelet"]
        )
        return self.loss_weight * total, terms

"""Paper-driven local implementation of SWIFT.

SWIFT maps each channel's sub-series with a single-level Haar wavelet
decomposition (`haar_dwt1d`, shared component), fuses the resulting
approximation/detail bands with a small shared depthwise convolution
("cross-band information fusion"), maps the fused coefficients from the
input's coefficient length to the horizon's coefficient length with a linear
or shallow MLP sub-series mapper, and reconstructs the forecast with the
matching inverse Haar transform. RevIN reversibly normalizes the input.
"""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.haar_dwt1d import HaarDWT1D, HaarIDWT1D
from tsflab.models._components.revin import RevIN


def _half_length(length: int) -> int:
    return (length + 1) // 2


class SubSeriesMapper(nn.Module):
    """Map input-length wavelet coefficients to horizon-length coefficients."""

    def __init__(self, in_len: int, out_len: int, hidden_size: int) -> None:
        super().__init__()
        if hidden_size > 0:
            self.net = nn.Sequential(
                nn.Linear(in_len, hidden_size), nn.ELU(), nn.Linear(hidden_size, out_len)
            )
        else:
            self.net = nn.Linear(in_len, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    """RevIN -> Haar DWT -> cross-band conv fusion -> sub-series mapping -> Haar IDWT."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int = 0,
        conv_kernel: int = 3,
        use_convdropout: bool = False,
        conv_dropout: float = 0.1,
        not_independent: bool = False,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 2:
            raise ValueError("seq_len and pred_len must each be >= 2")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.not_independent = not_independent
        self.use_convdropout = use_convdropout

        self.revin = RevIN(enc_in, affine=True)
        self.dwt = HaarDWT1D()
        self.idwt = HaarIDWT1D()
        self.fusion_conv = nn.Conv1d(
            in_channels=2, out_channels=2, kernel_size=conv_kernel,
            stride=1, padding=(conv_kernel - 1) // 2, padding_mode="zeros", bias=False,
        )
        if use_convdropout:
            self.conv_dropout = nn.Dropout(conv_dropout)

        in_len = _half_length(seq_len)
        out_len = _half_length(pred_len)
        if not_independent:
            self.mapper = nn.ModuleList(
                [SubSeriesMapper(in_len, out_len, hidden_size) for _ in range(enc_in)]
            )
        else:
            self.mapper = SubSeriesMapper(in_len, out_len, hidden_size)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        batch = x_enc.shape[0]
        normalized = self.revin(x_enc, "norm")
        values = normalized.permute(0, 2, 1)

        approx, detail = self.dwt(values)
        stacked = torch.stack([approx, detail], dim=-2)
        fused = self.fusion_conv(
            stacked.reshape(batch * self.enc_in, 2, stacked.shape[-1])
        ).reshape(batch, self.enc_in, 2, stacked.shape[-1]) + stacked
        if self.use_convdropout:
            fused = self.conv_dropout(fused)

        if self.not_independent:
            mapped = torch.stack(
                [self.mapper[i](fused[:, i, :, :]) for i in range(self.enc_in)], dim=1
            )
        else:
            mapped = self.mapper(fused)
        approx_out, detail_out = mapped[:, :, 0, :], mapped[:, :, 1, :]

        reconstructed = self.idwt(approx_out, detail_out, length=self.pred_len)
        forecast = reconstructed.permute(0, 2, 1)
        return self.revin(forecast, "denorm")

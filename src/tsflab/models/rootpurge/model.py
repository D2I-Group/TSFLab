"""Local RootPurge: bias-free linear forecasters with the root-purging objective.

Paper: Characteristic Root Analysis and Regularization for Linear Time Series
Forecasting (ICLR 2026), Section 4.2, Eq. (3)

    min_W ||Y_fut - G_W(Y_his)||^2 + lambda * ||G_W(P(Y_fut - G_W(Y_his)))||^2

``G_W`` is a linear map, either a time-domain weight matrix or the frequency
filter ``F^-1 o W_F o F``; ``P`` maps a horizon-length residual back to the
lookback length (zero padding when the horizon is shorter, cropping otherwise).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Mean-centred bias-free linear forecaster trained with root purging.

    ``forward`` is the plain forecast; the root-purging regularizer lives in
    ``purge_loss`` and is applied by the spec's training objective only.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        domain: str = "frequency",
        individual: bool = False,
        instance_norm: bool = True,
        purge_lambda: float = 0.25,
        purge_order: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.domain = domain
        self.individual = individual
        self.instance_norm = instance_norm
        self.purge_lambda = purge_lambda
        self.purge_order = purge_order
        groups = enc_in if individual else 1
        if domain == "time":
            # G_W(x) = W_T x with W_T in R^{H x L}, no bias (a linear, not affine, map).
            bound = 1.0 / math.sqrt(seq_len)
            self.weight = nn.Parameter(torch.empty(groups, pred_len, seq_len).uniform_(-bound, bound))
        else:
            # G_W = F^-1 o W_F o F with a complex W_F from L//2+1 to (L+H)//2+1 bins.
            self.total_len = seq_len + pred_len
            self.in_bins = seq_len // 2 + 1
            self.out_bins = self.total_len // 2 + 1
            bound = 1.0 / math.sqrt(self.in_bins)
            self.weight_real = nn.Parameter(torch.empty(groups, self.out_bins, self.in_bins).uniform_(-bound, bound))
            # Real-valued initial filter: the imaginary part starts at zero.
            self.weight_imag = nn.Parameter(torch.zeros(groups, self.out_bins, self.in_bins))

    @property
    def purge_scale(self) -> float:
        """lambda compensation ``max(L / H, 1) ** order`` for zero-padded residuals."""
        return max(self.seq_len / self.pred_len, 1.0) ** self.purge_order

    def linear_map(self, z: torch.Tensor) -> torch.Tensor:
        """Apply ``G_W`` to ``[B, L, C]`` and return ``[B, H, C]`` (no normalization)."""
        check_window(z, self.seq_len, self.enc_in)
        if self.domain == "time":
            weight = self.weight.expand(self.enc_in, -1, -1)
            return torch.einsum("blc,chl->bhc", z, weight)
        spectrum = torch.fft.rfft(z, dim=1)  # [B, L//2+1, C]
        real = self.weight_real.expand(self.enc_in, -1, -1)
        imag = self.weight_imag.expand(self.enc_in, -1, -1)
        out_real = torch.einsum("bfc,cof->boc", spectrum.real, real) - torch.einsum("bfc,cof->boc", spectrum.imag, imag)
        out_imag = torch.einsum("bfc,cof->boc", spectrum.real, imag) + torch.einsum("bfc,cof->boc", spectrum.imag, real)
        extended = torch.fft.irfft(torch.complex(out_real, out_imag), n=self.total_len, dim=1)
        # Energy compensation for the longer inverse transform, then keep the horizon.
        extended = extended * (self.total_len / self.seq_len)
        return extended[:, -self.pred_len :, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if not self.instance_norm:
            return self.linear_map(x_enc)
        mean = x_enc.mean(dim=1, keepdim=True)
        return self.linear_map(x_enc - mean) + mean

    def pad_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Operator ``P``: map a ``[B, H, C]`` residual to the lookback length ``L``.

        ``H > L`` keeps the first ``L`` steps; ``H < L`` places the residual in an
        all-zero window of length ``L`` at an offset drawn uniformly from
        ``0..L-H`` independently for every sample and channel.
        """
        batch, horizon, channels = residual.shape
        if horizon == self.seq_len:
            return residual
        if horizon > self.seq_len:
            return residual[:, : self.seq_len, :]
        start = torch.randint(0, self.seq_len - horizon + 1, (batch, 1, channels), device=residual.device)
        index = start + torch.arange(horizon, device=residual.device).view(1, horizon, 1)
        padded = residual.new_zeros(batch, self.seq_len, channels)
        return padded.scatter(1, index, residual)

    def purge_loss(self, residual: torch.Tensor) -> torch.Tensor:
        """Root-purging term ``lambda * c * mean(G_W(P(r))^2)`` with a stop-gradient on ``r``.

        ``purge_order > 1`` re-applies ``G_W o P`` to its own output (higher-order
        purging); ``c`` is ``purge_scale``. The zero target equals ``G_W(0)``
        because ``G_W`` is bias-free.
        """
        z = residual.detach()
        for _ in range(self.purge_order):
            z = self.linear_map(self.pad_residual(z))
        return self.purge_lambda * self.purge_scale * z.pow(2).mean()

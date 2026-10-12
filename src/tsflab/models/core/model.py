"""Clean-room Correction-space Interaction Refinement (CoRe).

CoRe adapts a frozen forecaster at test time. A base adapter produces per-variate
corrections ``Delta`` (COSA-style: a context-conditioned linear residual with a
tanh gate); CoRe then lets variates interact *in the correction space*, never in
the prediction space:

    alpha   = mean_c Delta_c                                   (paper Eq. 3)
    z_c     = [Delta_c || alpha]                               (Eq. 4)
    delta_c = W_up tanh(W_down z_c), rank r (default C)        (Eq. 5)
    g       = tanh(W_g s + b_g), s = [SE, LBR, MBR, HBR]       (Eqs. 7-8)
    Y       = Y_base + Delta + g * delta                       (Eq. 6)

TSFLab supplies a frozen last-value direct forecaster as a self-contained
default; callers may pass an external frozen forecast through ``base_forecast``.
Only the base adapter and the SCR/gate parameters are adaptable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.spectral_descriptor import SpectralDescriptor


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        context_len: int = 5,
        var_wise_gating: bool = True,
        scr_rank: int = 0,
        gate_init: float = 0.0,
        gate_bias_init: float = -1.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("CoRe dimensions must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.context_len = context_len
        self.var_wise_gating = var_wise_gating
        self.scr_rank = scr_rank or enc_in

        # Frozen last-value base forecaster (self-contained fallback carrier).
        self.base = ChannelWiseLinear(seq_len, pred_len, enc_in, individual=False)
        with torch.no_grad():
            self.base.linear.weight.zero_()
            self.base.linear.weight[:, -1] = 1.0
            self.base.linear.bias.zero_()
        self.base.requires_grad_(False)

        # Base adapter: Delta = tanh(g) * Linear([Y_base_c || context]).
        self.adapter = ChannelWiseLinear(
            pred_len + context_len, pred_len, enc_in, individual=var_wise_gating
        )
        for module in self.adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        self.gate = nn.Parameter(
            torch.full((enc_in if var_wise_gating else 1,), float(gate_init))
        )

        # Shared-anchor Correction Refinement bottleneck (Eq. 5), small init.
        self.scr_down = nn.Linear(2 * pred_len, self.scr_rank)
        self.scr_up = nn.Linear(self.scr_rank, pred_len)
        for layer in (self.scr_down, self.scr_up):
            nn.init.xavier_uniform_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

        # Input-conditioned spectral gate (Eqs. 7-8).
        self.spectral = SpectralDescriptor()
        self.scr_gate = nn.Linear(4, enc_in)
        nn.init.zeros_(self.scr_gate.weight)
        nn.init.constant_(self.scr_gate.bias, float(gate_bias_init))

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )

    def _fit_context(self, context: torch.Tensor, batch: int) -> torch.Tensor:
        """Return a ``[batch, context_len]`` context, left-padded by repetition."""
        if context.ndim == 3:
            context = context.mean(dim=2)
        if context.ndim != 2 or context.shape[0] != batch:
            raise ValueError("context must be [batch, length] or [batch, length, channels]")
        if context.shape[1] >= self.context_len:
            return context[:, -self.context_len :]
        pad = context[:, :1].expand(-1, self.context_len - context.shape[1])
        return torch.cat((pad, context), dim=1)

    def context_from_history(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback context: channel-mean of the latest observed steps."""
        return self._fit_context(x.mean(dim=2), x.shape[0])

    def base_correction(self, base_forecast: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """COSA-style base adapter: ``Delta = tanh(g) * W[Y_base_c || context]``."""
        batch = base_forecast.shape[0]
        context = self._fit_context(context, batch)
        y = base_forecast.transpose(1, 2)  # [B, C, H]
        augmented = torch.cat((y, context[:, None, :].expand(-1, self.enc_in, -1)), dim=-1)
        delta = self.adapter(augmented) * self.gate.tanh().view(1, -1, 1)
        return delta.transpose(1, 2)  # [B, H, C]

    def refine(self, delta: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        """Shared-anchor correction refinement gated by the spectral descriptor."""
        corr = delta.transpose(1, 2)  # [B, C, H]
        anchor = corr.mean(dim=1, keepdim=True).expand_as(corr)  # Eq. 3
        z = torch.cat((corr, anchor), dim=-1)  # Eq. 4
        refinement = self.scr_up(torch.tanh(self.scr_down(z)))  # Eq. 5
        gate = torch.tanh(self.scr_gate(spectral)).unsqueeze(-1)  # Eq. 8, [B, C, 1]
        return (corr + gate * refinement).transpose(1, 2)  # Eq. 6

    def correct(
        self, base_forecast: torch.Tensor, context: torch.Tensor, spectral: torch.Tensor
    ) -> torch.Tensor:
        if base_forecast.ndim != 3 or base_forecast.shape[1:] != (self.pred_len, self.enc_in):
            raise ValueError("base_forecast does not match CoRe's output contract")
        if spectral.shape != (base_forecast.shape[0], 4):
            raise ValueError("spectral descriptor must be [batch, 4]")
        # The backbone carrier is frozen even if the caller's tensor tracks gradients.
        base_forecast = base_forecast.detach()
        delta = self.base_correction(base_forecast, context)
        return base_forecast + self.refine(delta, spectral)

    def forecast_with_context(
        self,
        x_enc: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        base_forecast: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate(x_enc)
        if base_forecast is None:
            base_forecast = self.base(x_enc.transpose(1, 2)).transpose(1, 2)
        if context is None:
            context = self.context_from_history(x_enc)
        return self.correct(base_forecast, context, self.spectral(x_enc))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        return self.forecast_with_context(x_enc)

    def adaptable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Base-adapter plus CoRe parameters updated during leakage-free TTA."""
        return (
            *self.adapter.parameters(),
            self.gate,
            *self.scr_down.parameters(),
            *self.scr_up.parameters(),
            *self.scr_gate.parameters(),
        )

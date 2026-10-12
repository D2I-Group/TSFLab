"""Clean-room Context-aware Output-Space Adapter (COSA).

The paper defines COSA around a frozen base forecast. TSFLab supplies a
frozen last-value direct forecaster for a self-contained default, while callers
may pass forecasts from another model through ``base_forecast=...``. Only the
linear residual and bounded gate are trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        context_len: int = 10,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.context_len = context_len

        self.base = ChannelWiseLinear(seq_len, pred_len, enc_in, individual=False)
        with torch.no_grad():
            self.base.linear.weight.zero_()
            self.base.linear.weight[:, -1] = 1.0
            self.base.linear.bias.zero_()
        self.base.requires_grad_(False)
        self.residual = nn.Linear(pred_len + context_len, pred_len)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _validate(self, x: torch.Tensor) -> None:
        check_window(x, self.seq_len, self.enc_in)

    def context_from_history(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback context when a revealed-target mean buffer is unavailable."""
        history = x.transpose(1, 2)
        if history.shape[-1] >= self.context_len:
            return history[..., -self.context_len :]
        return torch.cat(
            (
                history[..., :1].expand(-1, -1, self.context_len - history.shape[-1]),
                history,
            ),
            dim=-1,
        )

    def _prepare_context(self, context: torch.Tensor, batch: int) -> torch.Tensor:
        if context.ndim == 2:
            if context.shape[0] != batch:
                raise ValueError("context batch dimension is invalid")
            context = context[:, None, :].expand(-1, self.enc_in, -1)
        elif context.ndim == 3:
            if context.shape[0] != batch:
                raise ValueError("context batch dimension is invalid")
            if context.shape[1] == self.context_len and context.shape[2] == self.enc_in:
                context = context.transpose(1, 2)
            elif context.shape[1] != self.enc_in:
                raise ValueError("context must be [batch, context, channels]")
        else:
            raise ValueError("context must be rank two or three")
        if context.shape[-1] >= self.context_len:
            return context[..., -self.context_len :]
        return torch.cat(
            (
                context[..., :1].expand(-1, -1, self.context_len - context.shape[-1]),
                context,
            ),
            dim=-1,
        )

    def correct(self, base_forecast: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Paper equation: y_hat = y0 + tanh(g) W[y0 || context]."""
        if base_forecast.ndim != 3 or base_forecast.shape[1:] != (
            self.pred_len,
            self.enc_in,
        ):
            raise ValueError("base_forecast does not match COSA's output contract")
        # COSA adapts only its output adapter; an externally supplied carrier is
        # therefore treated as frozen even if its tensor still tracks gradients.
        base_forecast = base_forecast.detach()
        context = self._prepare_context(context, base_forecast.shape[0])
        augmented = torch.cat((base_forecast.transpose(1, 2), context), dim=-1)
        correction = self.residual(augmented)
        return (base_forecast.transpose(1, 2) + self.gate.tanh() * correction).transpose(
            1, 2
        )

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
        return self.correct(base_forecast, context)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        return self.forecast_with_context(x_enc)

    def adaptable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the adapter-only parameter set used for leakage-free TTA."""
        return tuple(self.residual.parameters()) + (self.gate,)

"""Local CosDir: a direction-aware training loss on a DLinear carrier.

CosDir changes the training objective, not the architecture. ``forward`` is an
ordinary DLinear point forecast; ``training_objective`` adds the cosine penalty
between horizon difference vectors (paper Eq. 2) or its uncertainty-weighted
form CosDir-UW (Eq. 3).
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._inputs import check_window

Weighting = Literal["fixed", "uw"]


def horizon_differences(series: torch.Tensor, last_observed: torch.Tensor) -> torch.Tensor:
    """First differences along the horizon, step 0 taken against ``x_L``.

    ``series`` is ``[B, H, C]`` and ``last_observed`` ``[B, 1, C]``;
    Delta y_h = y_h - y_{h-1} with y_0 = x_L (Preliminaries, Algorithm 1 l.1-2).
    """
    if series.ndim != 3 or last_observed.shape != (series.shape[0], 1, series.shape[2]):
        raise ValueError("last_observed must be [batch, 1, channels] matching the series")
    previous = torch.cat((last_observed, series[:, :-1]), dim=1)
    return series - previous


def direction_penalty(
    forecast: torch.Tensor,
    target: torch.Tensor,
    last_observed: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_dir: mean over samples and channels of 1 - cos(Delta y_hat_c, Delta y_c).

    The cosine is taken over the horizon with the paper's stabilised
    denominator ||Delta y_hat|| ||Delta y|| + eps (Eq. 2, Algorithm 1 l.4-6).
    """
    if forecast.shape != target.shape:
        raise ValueError("forecast and target must have the same shape")
    d_forecast = horizon_differences(forecast, last_observed)
    d_target = horizon_differences(target, last_observed)
    inner = (d_forecast * d_target).sum(dim=1)
    norms = d_forecast.norm(dim=1) * d_target.norm(dim=1)
    cosine = inner / (norms + eps)
    return (1.0 - cosine).mean()


class Model(nn.Module):
    """DLinear forecaster trained with the CosDir or CosDir-UW objective."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_size: int = 25,
        individual: bool = False,
        weighting: Weighting = "fixed",
        dir_lambda: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.weighting = weighting
        self.dir_lambda = float(dir_lambda)
        self.eps = float(eps)

        self.backbone = DLinearBackbone(
            enc_in, seq_len, pred_len, kernel_size=kernel_size, individual=individual
        )
        self._uniform_projection_init()
        # CosDir-UW log-variances (s1 for the base loss, s2 for L_dir), Eq. (3).
        if weighting == "uw":
            self.log_variances = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("log_variances", None)

    def _uniform_projection_init(self) -> None:
        """Backbone default of the benchmark code: projection weights start at 1/L."""
        for projection in (self.backbone.seasonal_projection, self.backbone.trend_projection):
            linears = projection.linears if projection.individual else [projection.linear]
            for linear in linears:
                nn.init.constant_(linear.weight, 1.0 / self.seq_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        return self.backbone(x_enc)

    def combine(self, base_loss: torch.Tensor, dir_loss: torch.Tensor) -> torch.Tensor:
        """Eq. (2) L_base + lambda L_dir, or Eq. (3) e^-s1 L_base + e^-s2 L_dir + (s1+s2)/2."""
        if self.weighting == "fixed":
            return base_loss + self.dir_lambda * dir_loss
        s1, s2 = self.log_variances[0], self.log_variances[1]
        return torch.exp(-s1) * base_loss + torch.exp(-s2) * dir_loss + 0.5 * (s1 + s2)

    def effective_lambda(self) -> float:
        """lambda_eff = e^(s1 - s2) for CosDir-UW; the fixed weight otherwise."""
        if self.weighting == "fixed":
            return self.dir_lambda
        s1, s2 = self.log_variances.detach()
        return float(torch.exp(s1 - s2))

    def training_objective(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        last_observed: torch.Tensor,
        base_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        align: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forecast ``x`` and return (forecast, total loss, parts).

        ``align`` selects the target channels of the forecast (``MS`` runs);
        ``last_observed`` is the matching ``[B, 1, C_target]`` last input step.
        """
        forecast = self(x)
        aligned = align(forecast) if align is not None else forecast
        base = base_criterion(aligned, target)
        direction = direction_penalty(aligned, target, last_observed, self.eps)
        return forecast, self.combine(base, direction), {"base": base, "direction": direction}

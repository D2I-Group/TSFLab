"""Clean-room GOTSF implementation from the interval-policy equations.

GOTSF is a training and inference policy rather than a new backbone. This
module uses a small channel-independent forecaster and implements the paper's
defining pieces directly: discrete interval conditions, soft boundary decay,
an interval-membership head, and confidence-weighted patching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Goal-conditioned direct forecaster using the paper's D*_L policy."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        num_intervals: int = 4,
        interval_min: float = -2.0,
        interval_max: float = 2.0,
        decay_rate: float = 50.0,
        classification_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_intervals = num_intervals
        self.decay_rate = float(decay_rate)
        self.classification_weight = float(classification_weight)

        self.register_buffer(
            "interval_edges", torch.linspace(interval_min, interval_max, num_intervals + 1)
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.interval_encoder = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.regression_head = nn.Linear(d_model, pred_len)
        self.classification_head = nn.Linear(d_model, pred_len)
        self.last_interval_predictions: torch.Tensor | None = None
        self.last_interval_confidences: torch.Tensor | None = None

    @property
    def interval_bounds(self) -> torch.Tensor:
        """Return the disjoint support of the discrete interval policy."""
        return torch.stack((self.interval_edges[:-1], self.interval_edges[1:]), -1)

    def decay(self, target: torch.Tensor, interval_index: int) -> torch.Tensor:
        """Evaluate Eq. (8), ``exp(-nu max(0, |y-mid|-half_width))``."""
        if not 0 <= interval_index < self.num_intervals:
            raise IndexError("interval_index is outside the discrete policy support")
        low, high = self.interval_bounds[interval_index]
        midpoint = (low + high) / 2
        half_width = (high - low) / 2
        return torch.exp(-self.decay_rate * torch.relu((target - midpoint).abs() - half_width))

    def intersecting_intervals(
        self, target_interval: tuple[float, float] | torch.Tensor | None
    ) -> torch.Tensor:
        """Implement Eq. (13), returning bins intersecting a query interval."""
        if target_interval is None:
            return torch.ones(
                self.num_intervals, dtype=torch.bool, device=self.interval_edges.device
            )
        query = torch.as_tensor(
            target_interval, dtype=self.interval_edges.dtype, device=self.interval_edges.device
        )
        if query.shape != (2,) or bool(query[1] < query[0]):
            raise ValueError("target_interval must be an ordered pair [minimum, maximum]")
        bounds = self.interval_bounds
        selected = (bounds[:, 1] >= query[0]) & (bounds[:, 0] <= query[1])
        if not bool(selected.any()):
            raise ValueError("target_interval does not intersect the configured forecasting space")
        return selected

    def interval_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return regression values and membership confidences for every interval."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )
        history = self.history_encoder(x.transpose(1, 2))
        bounds = self.interval_bounds
        condition = torch.stack(
            ((bounds[:, 0] + bounds[:, 1]) / 2, (bounds[:, 1] - bounds[:, 0]) / 2), -1
        )
        hidden = history.unsqueeze(2) + self.interval_encoder(condition).view(
            1, 1, self.num_intervals, -1
        )
        predictions = self.regression_head(F.gelu(hidden)).permute(0, 2, 3, 1)
        confidence = torch.sigmoid(self.classification_head(F.gelu(hidden))).permute(
            0, 2, 3, 1
        )
        return predictions, confidence

    def goal_oriented_loss(
        self, x: torch.Tensor, target: torch.Tensor, interval_index: int,
        outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Regression-plus-membership objective corresponding to Eqs. (9)--(12)."""
        if target.shape != (x.shape[0], self.pred_len, self.enc_in):
            raise ValueError("target has the wrong forecasting shape")
        predictions, confidence = outputs or self.interval_outputs(x)
        prediction = predictions[:, interval_index]
        membership = (
            (target >= self.interval_bounds[interval_index, 0])
            & (target <= self.interval_bounds[interval_index, 1])
        ).to(target.dtype)
        # Official trainer: element-wise decay weights on the L1 regression term
        # and an unweighted mean BCE on the membership term.
        regression = ((prediction - target).abs() * self.decay(target, interval_index)).mean()
        classification = F.binary_cross_entropy(
            confidence[:, interval_index].clamp(1e-6, 1 - 1e-6), membership
        )
        return regression + self.classification_weight * classification

    def goal_oriented_objective(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training pass: mean per-interval loss and the confidence-weighted forecast."""
        outputs = self.interval_outputs(x)
        loss = torch.stack(
            [
                self.goal_oriented_loss(x, target, index, outputs)
                for index in range(self.num_intervals)
            ]
        ).mean()
        predictions, confidence = outputs
        forecast = (predictions * confidence).sum(1) / confidence.sum(1).clamp_min(1e-6)
        return forecast, loss

    def forecast_interval(
        self,
        x_enc: torch.Tensor,
        target_interval: tuple[float, float] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        predictions, confidence = self.interval_outputs(x_enc)
        selected = self.intersecting_intervals(target_interval).view(1, -1, 1, 1)
        weights = confidence * selected
        output = (predictions * weights).sum(1) / weights.sum(1).clamp_min(1e-6)
        self.last_interval_predictions = predictions
        self.last_interval_confidences = confidence
        return output

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        return self.forecast_interval(x_enc)

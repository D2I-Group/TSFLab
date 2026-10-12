"""Clean-room Time-o1 transformed-label alignment objective and local backbone.

Time-o1 is a model-agnostic training objective rather than a forecasting
architecture. This module therefore supplies a small independent temporal
backbone plus the paper's per-variate SVD basis fitting and Equations (4)--(5)
for use by experiment runners.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        alpha: float = 0.8,
        rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )
        self.skip = nn.Linear(seq_len, pred_len)
        identity = torch.eye(pred_len).expand(enc_in, -1, -1).clone()
        self.register_buffer("projection", identity)
        self.register_buffer("projection_ready", torch.tensor(False))

    @torch.no_grad()
    def fit_projection(self, labels: torch.Tensor) -> torch.Tensor:
        """Fit per-variate right-singular-vector bases from training labels."""
        if (
            labels.ndim != 3
            or labels.shape[1] != self.pred_len
            or not 1 <= labels.shape[2] <= self.enc_in
        ):
            raise ValueError(
                f"expected labels (N, {self.pred_len}, <={self.enc_in}), got {tuple(labels.shape)}"
            )
        centered = labels - labels.mean(dim=1, keepdim=True)
        standardized = centered / torch.sqrt(
            centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        bases = []
        channels = labels.shape[2]
        for channel in range(channels):
            _, _, right = torch.linalg.svd(standardized[:, :, channel], full_matrices=True)
            bases.append(right.transpose(0, 1))
        # Fewer label channels (``MS`` targets) fit the trailing channels.
        self.projection[-channels:] = torch.stack(bases)
        self.projection_ready.fill_(True)
        return self.projection

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Project horizon values into descending-significance models._components."""
        if (
            values.ndim != 3
            or values.shape[1] != self.pred_len
            or not 1 <= values.shape[2] <= self.enc_in
        ):
            raise ValueError(
                f"expected values (B, {self.pred_len}, <={self.enc_in}), got {tuple(values.shape)}"
            )
        return torch.einsum("btc,ctk->bkc", values, self.projection[-values.shape[2]:])

    def transformed_alignment_loss(
        self,
        forecast: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "sum",
        temporal=None,
    ) -> torch.Tensor:
        """Compute paper Equation (5) using the retained leading components.

        ``reduction="sum"`` is the equation as written. ``"mean"`` (used by the
        runner objective, like the official trainer) averages both terms so the
        loss scale does not grow with batch size; ``temporal`` then replaces the
        squared-error term with the configured criterion.
        """
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        if not bool(self.projection_ready):
            raise RuntimeError("fit_projection must be called on training labels first")
        forecast_components = self.transform(forecast)
        target_components = self.transform(target)
        retained = max(1, round(self.rank_ratio * self.pred_len))
        difference = (
            forecast_components[:, :retained] - target_components[:, :retained]
        ).abs()
        if reduction == "sum":
            transformed, temporal_loss = difference.sum(), (forecast - target).square().sum()
        else:
            transformed = difference.mean()
            temporal_loss = (
                temporal(forecast, target)
                if temporal is not None
                else (forecast - target).square().mean()
            )
        return self.alpha * transformed + (1 - self.alpha) * temporal_loss

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        values = x_enc.transpose(1, 2)
        return (self.temporal(values) + self.skip(values)).transpose(1, 2)


__all__ = ["Model"]

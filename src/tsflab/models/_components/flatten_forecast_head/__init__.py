"""Linear forecasting head for flattened feature and patch axes."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlattenForecastHead(nn.Module):
    """Map ``(B, C, D, P)``-like inputs to ``(B, C, horizon)``.

    The final two axes are flattened in their existing order. With
    ``individual=True`` each channel owns separate projection parameters;
    otherwise one projection is shared across all channels.
    """

    def __init__(
        self,
        individual: bool,
        n_vars: int,
        nf: int,
        target_window: int,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if individual:
            self.flattens = nn.ModuleList(nn.Flatten(start_dim=-2) for _ in range(n_vars))
            self.linears = nn.ModuleList(nn.Linear(nf, target_window) for _ in range(n_vars))
            self.dropouts = nn.ModuleList(nn.Dropout(head_dropout) for _ in range(n_vars))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the last two axes and project to the forecast horizon."""
        if not self.individual:
            return self.dropout(self.linear(self.flatten(x)))
        outputs = [
            self.dropouts[index](self.linears[index](self.flattens[index](x[:, index, :, :])))
            for index in range(self.n_vars)
        ]
        return torch.stack(outputs, dim=1)


__all__ = ["FlattenForecastHead"]

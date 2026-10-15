"""Independent soft nearest-reference forecasting baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Kernel-weight trainable reference windows and their future continuations."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_prototypes: int = 32,
        kernel_gamma: float = 0.08,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.kernel_gamma = kernel_gamma
        self.reference_windows = nn.Parameter(torch.empty(num_prototypes, seq_len, enc_in))
        self.reference_futures = nn.Parameter(torch.empty(num_prototypes, pred_len, enc_in))
        nn.init.normal_(self.reference_windows, std=0.02)
        nn.init.normal_(self.reference_futures, std=0.02)
        self.aux_loss: torch.Tensor | None = None

    def neighbor_weights(self, x: torch.Tensor) -> torch.Tensor:
        squared_distance = (x.unsqueeze(1) - self.reference_windows.unsqueeze(0)).square().mean(dim=(-1, -2))
        return torch.softmax(-self.kernel_gamma * squared_distance, dim=-1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        weights = self.neighbor_weights(x_enc)
        forecast = torch.einsum("bk,khc->bhc", weights, self.reference_futures)
        self.aux_loss = forecast.new_zeros(())
        return forecast

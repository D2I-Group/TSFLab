"""GaussianMLP: a flatten-MLP Gaussian distribution forecaster.

Flattens the input window and maps it through an MLP to per-(horizon, channel)
Gaussian parameters (loc, scale > 0), emitting a rank-4 (B, pred_len, C, 2)
tensor consumed by the Phase 1 nll_gaussian loss and Gaussian CRPS metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.gaussian_parameter_head import GaussianParameterHead
from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.eps = eps
        # Probabilistic output contract (Phase 1): emit (loc, scale) per element.
        self.output_type = "distribution"
        self.distribution_family = "gaussian"

        in_dim = seq_len * enc_in
        out_dim = pred_len * self.c_out
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.parameter_head = GaussianParameterHead(prev, out_dim, eps=eps)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        B = x_enc.shape[0]
        h = self.backbone(x_enc.reshape(B, -1))  # (B, hidden)
        loc, scale = self.parameter_head(h)
        loc = loc.reshape(B, self.pred_len, self.c_out)
        scale = scale.reshape(B, self.pred_len, self.c_out)
        return torch.stack([loc, scale], dim=-1)  # (B, pred_len, c_out, 2)

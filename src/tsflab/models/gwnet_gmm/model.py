"""Graph WaveNet with a Gaussian-mixture prediction layer (Xiong et al., 2026).

Independent implementation of the GMM_Pred layer of "Unveiling Stochasticity:
Universal Multi-modal Probabilistic Modeling for Traffic Forecasting"
(arXiv 2604.16084, Sec. 3.1-3.3) on the Graph WaveNet backbone of the paper's
headline configuration. Omissions were resolved against the official repository
(Weijiang-Xiong/OpenSkyTraffic, revision 824534a112b37973481eedaa705d1a14279f0743:
``skytraffic/models/layers/gmmpred.py``, ``gwnet.py``, ``gwnet_gmm.py`` and
``config/gwnet/GWNET_GMM.py``); that repository has no license, so nothing was
copied or imported.

Eq. (3): Z = ST_Backbone(X), p(X_future) = GMM_Pred(Z), with per element
    Z'     = Dropout(ReLU(LayerNorm(Linear(Z))))                      Eq. (6)
    pi     = Softmax_K(Linear(Z'))                                    Eq. (7)-(8)
    mu     = r + Linear(Z') * s * (1 + s_learned)                     Eq. (7)
    log s2 = Linear(Z')                                               Eq. (7)
    -log p(x) = -logsumexp_k[log pi_k - (x - mu_k)^2 / (2 s2_k) - log(2 pi s2_k) / 2]   Eq. (5)
    x_hat  = sum_k pi_k mu_k                                          Eq. (10)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.diffusion_conv import DiffusionConv2d
from tsflab.models._components.marks import to_spatiotemporal

_LOG_2PI = math.log(2.0 * math.pi)


class GaussianMixtureHead(nn.Module):
    """GMM_Pred: per-location mixing weights, anchored means, and log-variances."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        pred_len: int,
        anchors: tuple[float, ...],
        spacing: float,
        dropout: float,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.num_components = len(anchors)
        # Reference values r and spacing s are fixed priors (Fig. 3, Eq. 9).
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.register_buffer(
            "spacing", torch.full((len(anchors),), float(spacing), dtype=torch.float32)
        )
        # Official code: a learnable relative correction of the spacing, initialized to zero.
        self.spacing_scale = nn.Parameter(torch.zeros(len(anchors)))
        self.projection = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        width = pred_len * self.num_components
        self.mixing = nn.Linear(hidden_dim, width)
        self.offset = nn.Linear(hidden_dim, width)
        self.log_variance = nn.Linear(hidden_dim, width)
        if zero_init:
            self.reset_to_prior()

    def reset_to_prior(self) -> None:
        """Weakly informative prior: equal weights, means at r, unit variances (Sec. 3.2)."""
        for layer in (self.mixing, self.offset, self.log_variance):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.mixing.bias, 1.0 / self.num_components)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, N, D]`` location features -> three ``[B, pred_len, N, K]`` tensors."""
        hidden = self.dropout(F.relu(self.norm(self.projection(features))))
        batch, nodes, _ = hidden.shape

        def per_step(layer: nn.Linear) -> torch.Tensor:
            out = layer(hidden).view(batch, nodes, self.pred_len, self.num_components)
            return out.transpose(1, 2)

        weights = torch.softmax(per_step(self.mixing), dim=-1)
        means = self.anchors + per_step(self.offset) * self.spacing * (1.0 + self.spacing_scale)
        return weights, means, per_step(self.log_variance)


def gmm_negative_log_likelihood(
    weights: torch.Tensor, means: torch.Tensor, log_variance: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Eq. (5) averaged over every finite target element (NaN marks a missing label)."""
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        raise ValueError("GMM negative log-likelihood needs at least one finite target")
    x = target[valid].unsqueeze(-1)
    log_weights = torch.log(weights[valid])
    mu, log_var = means[valid], log_variance[valid]
    component = log_weights - 0.5 * (x - mu).pow(2) * torch.exp(-log_var) - 0.5 * (log_var + _LOG_2PI)
    return -torch.logsumexp(component, dim=-1).mean()


class WaveNetLayer(nn.Module):
    """Valid dilated gated convolution, adaptive-graph diffusion, residual, BatchNorm."""

    def __init__(self, channels: int, skip_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.filter = nn.Conv2d(channels, channels, (1, kernel_size), dilation=(1, dilation))
        self.gate = nn.Conv2d(channels, channels, (1, kernel_size), dilation=(1, dilation))
        self.skip = nn.Conv2d(channels, skip_channels, 1)
        self.graph = DiffusionConv2d(channels, channels, dropout, support_len=1, order=2)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # No padding: each layer shortens the time axis by dilation * (kernel_size - 1).
        gated = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        mixed = self.graph(gated, [support]) + x[..., -gated.shape[-1] :]
        return self.norm(mixed), self.skip(gated)


class Model(nn.Module):
    """GWNet backbone (adaptive adjacency only) with the paper's GMM output layer."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        in_dim: int = 2,
        dropout: float = 0.3,
        hidden_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        node_embedding_dim: int = 10,
        anchors: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
        spacing: float = 1.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, num_nodes) < 1:
            raise ValueError("GWNetGMM lengths and num_nodes must be positive")
        receptive = 1
        dilations = []
        for _ in range(blocks):
            step = kernel_size - 1
            for layer in range(layers):
                dilations.append(2**layer)
                receptive += step
                step *= 2
        # The input is left padded by one step, so the stack must collapse seq_len + 1 to one step.
        if seq_len + 1 > receptive:
            raise ValueError(
                f"seq_len + 1 = {seq_len + 1} exceeds the receptive field {receptive}; "
                "increase blocks or layers"
            )
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.in_dim, self.receptive_field = in_dim, receptive
        self.source_nodes = nn.Parameter(torch.randn(num_nodes, node_embedding_dim))
        self.target_nodes = nn.Parameter(torch.randn(node_embedding_dim, num_nodes))
        self.start = nn.Conv2d(in_dim, hidden_channels, 1)
        self.layers = nn.ModuleList(
            WaveNetLayer(hidden_channels, skip_channels, kernel_size, dilation, dropout)
            for dilation in dilations
        )
        self.end = nn.Conv2d(skip_channels, end_channels, 1)
        self.head = GaussianMixtureHead(
            end_channels, end_channels, pred_len, tuple(anchors), spacing, dropout
        )

    def backbone(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """ST_Backbone of Eq. (3): ``[B, seq_len, N]`` -> location features ``[B, N, end_channels]``."""
        if x_enc.ndim != 3 or tuple(x_enc.shape[1:]) != (self.seq_len, self.num_nodes):
            raise ValueError(f"GWNetGMM expects (B, {self.seq_len}, {self.num_nodes}) values")
        data = to_spatiotemporal(x_enc, x_mark_enc)
        if data.shape[-1] < self.in_dim:
            raise ValueError("GWNetGMM received fewer input features than in_dim")
        x = data[..., : self.in_dim].permute(0, 3, 2, 1)  # [B, F, N, T]
        x = F.pad(x, (1 + self.receptive_field - (self.seq_len + 1), 0))
        x = self.start(x)
        support = adaptive_node_embedding_adjacency(self.source_nodes, self.target_nodes)
        skip = None
        for layer in self.layers:
            x, s = layer(x, support)
            skip = s if skip is None else s + skip[..., -s.shape[-1] :]
        features = F.relu(self.end(F.relu(skip)))  # [B, end_channels, N, 1]
        return features.squeeze(-1).transpose(1, 2)

    def mixture(self, x_enc, x_mark_enc=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicted mixture ``(weights, means, log_variance)``, each ``[B, pred_len, N, K]``."""
        return self.head(self.backbone(x_enc, x_mark_enc))

    @staticmethod
    def point_forecast(weights: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """Eq. (10): the mixture mean ``sum_k pi_k mu_k``."""
        return (weights * means).sum(dim=-1)

    def training_objective(self, x_enc, x_mark_enc, target):
        """Return the point forecast and the GMM NLL of Eq. (5) from one forward pass."""
        weights, means, log_variance = self.mixture(x_enc, x_mark_enc)
        return self.point_forecast(weights, means), gmm_negative_log_likelihood(
            weights, means, log_variance, target
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        weights, means, _ = self.mixture(x_enc, x_mark_enc)
        return self.point_forecast(weights, means)


__all__ = ["GaussianMixtureHead", "Model", "WaveNetLayer", "gmm_negative_log_likelihood"]

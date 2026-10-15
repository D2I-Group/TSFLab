"""RI-Loss: residual-informed HSIC training objective on a DLinear forecaster (AAAI 2026).

RI-Loss changes how a forecaster is trained, not how it forecasts. With residual
``R = Y - Y_hat`` and a fresh random series ``eps`` of the same shape, the loss is
(paper Eq. 8, Algorithm 1)

    L = L_obs(Y_hat, Y) + lambda * exp(-tau * HSIC(R, eps)),

so minimizing it *maximizes* the kernel dependence between the residual and pure
noise: whatever the model leaves unexplained should look like noise. HSIC uses
Gaussian kernels and the centred-trace (biased) estimator ``tr(K H L H) / (n - 1)^2``.
As in the official code, the ``n`` kernel samples of one forecast window are its
channels (each a length-``H`` residual vector), and the per-window statistics are
summed over the batch before the exponential.

The forecaster is the shared DLinear backbone (one of the paper's backbones); the
objective is exposed through :meth:`Model.training_objective`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._inputs import check_window


def gaussian_gram(samples: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """``[B, n, d] -> [B, n, n]`` with entries ``exp(-||s_i - s_j||^2 / bandwidth)``."""
    sq_norm = samples.square().sum(dim=-1, keepdim=True)
    sq_dist = sq_norm + sq_norm.transpose(-2, -1) - 2.0 * samples @ samples.transpose(-2, -1)
    return torch.exp(-sq_dist / bandwidth)


def hsic(first: torch.Tensor, second: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """Biased empirical HSIC per batch element, ``[B, n, d] x [B, n, d'] -> [B]``.

    ``HSIC = tr(K H L H) / (n - 1)^2`` with ``H = I - 11^T / n`` and Gaussian Gram
    matrices ``K`` (first) and ``L`` (second) of the same bandwidth.
    """
    if first.shape[:2] != second.shape[:2]:
        raise ValueError("HSIC inputs must share batch and sample axes")
    n = first.shape[1]
    if n < 2:
        raise ValueError("HSIC needs at least two samples (channels) per window")
    k = gaussian_gram(first, bandwidth)
    l_gram = gaussian_gram(second, bandwidth)
    centring = torch.eye(n, dtype=k.dtype, device=k.device) - 1.0 / n
    centred_k = centring @ k @ centring
    return (l_gram * centred_k).sum(dim=(-2, -1)) / float((n - 1) ** 2)


class Model(nn.Module):
    """DLinear forecaster carrying the RI-Loss training objective."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_size: int = 25,
        individual: bool = False,
        ri_weight: float = 10.0,
        temperature: float = 1.0,
        bandwidth: float = 1.0,
        noise_low: float = 0.0,
        noise_high: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.ri_weight = ri_weight
        self.temperature = temperature
        self.bandwidth = bandwidth
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.backbone = DLinearBackbone(enc_in, seq_len, pred_len, kernel_size, individual)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        return self.backbone(x_enc)

    def sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Uniform noise on ``[noise_low, noise_high)`` with the residual's shape."""
        return torch.rand_like(like) * (self.noise_high - self.noise_low) + self.noise_low

    def residual_dependence(self, residual: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Summed per-window HSIC between ``[B, H, C]`` residuals and noise (channels as samples)."""
        return hsic(residual.transpose(1, 2).contiguous(), noise.transpose(1, 2).contiguous(), self.bandwidth).sum()

    def ri_loss(
        self,
        forecast: torch.Tensor,
        target: torch.Tensor,
        observation_loss: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Eq. (8): ``L_obs + ri_weight * exp(-temperature * HSIC(target - forecast, noise))``.

        ``observation_loss`` defaults to the MSE of the paper; ``noise`` is drawn fresh
        from the uniform distribution when not given.
        """
        if forecast.shape != target.shape:
            raise ValueError("forecast and target must have the same shape")
        residual = target - forecast
        noise = self.sample_noise(residual) if noise is None else noise
        if observation_loss is None:
            observation_loss = F.mse_loss(forecast, target)
        dependence = self.residual_dependence(residual, noise)
        return observation_loss + self.ri_weight * torch.exp(-self.temperature * dependence)

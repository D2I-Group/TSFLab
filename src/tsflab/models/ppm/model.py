"""PPM: parametric prior mapping for probabilistic forecasting (Li et al., ICML 2026).

A channel-independent MLP encoder maps each normalized history to the mean and
log-variance of a diagonal Gaussian latent prior (Eqs. 2-3). Reparameterized
latent draws (Eq. 4) are pushed through a two-layer GELU MLP to forecast samples
(Eq. 5). Training minimizes ``alpha * KDE-NLL + MSE(sample mean)`` (Eqs. 6-10);
the forecast is the set of empirical quantiles of the samples.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.empirical_quantiles import empirical_quantiles


def kde_nll(samples: torch.Tensor, target: torch.Tensor, bandwidth: float, log_floor: float = -25.0) -> torch.Tensor:
    """Gaussian-kernel KDE negative log-likelihood (Eqs. 6-8) in the log domain.

    ``samples`` is ``[B, K, L, C]`` and ``target`` ``[B, L, C]``. The per-element
    log density ``logsumexp_k(-u^2 / 2) - log(sqrt(2 pi) K h)`` is averaged over
    the horizon and channels of each window, that average is floored at
    ``log_floor`` (the official code's ``-25``), and the batch mean is negated.
    """
    count = samples.shape[1]
    scaled = (samples - target.unsqueeze(1)) / bandwidth
    log_kernel = -0.5 * scaled.square() - 0.5 * math.log(2 * math.pi)
    log_density = torch.logsumexp(log_kernel, dim=1) - (math.log(count) + math.log(bandwidth))
    per_window = log_density.mean(dim=(-1, -2)).clamp(min=log_floor)
    return -per_window.mean()


def mean_matching(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. (9): squared error of the sample mean, averaged over every element."""
    return (samples.mean(dim=1) - target).square().mean()


class Model(nn.Module):
    """``forward`` returns empirical quantiles ``[B, pred_len, enc_in, Q]`` of ``num_samples`` draws."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        quantile_levels: list[float],
        d_model: int = 256,
        d_ff: int = 512,
        num_samples: int = 100,
        bandwidth: float = 0.3,
        alpha: float = 0.1,
        log_floor: float = -25.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        levels = list(quantile_levels)
        if not levels or any(not 0.0 < q < 1.0 for q in levels) or levels != sorted(set(levels)):
            raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
        self.output_type = "quantile"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_samples = num_samples
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.log_floor = log_floor
        self.eps = eps
        self.register_buffer("quantile_levels", torch.tensor(levels), persistent=False)

        # Encoder f_theta (Eq. 2): one GELU hidden layer, then mean and log-variance heads.
        self.encoder = nn.Linear(seq_len, d_model)
        self.mean_head = nn.Linear(d_model, d_model)
        self.logvar_head = nn.Linear(d_model, d_model)
        # Mapping g_phi (Eq. 5): the official code conditions it on a history embedding.
        self.history = nn.Linear(seq_len, d_model)
        self.mapping = nn.Sequential(nn.Linear(2 * d_model, d_ff), nn.GELU(), nn.Linear(d_ff, pred_len))

    def prior(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. (2): ``[N, C, L] -> (mu, log sigma^2)``, each ``[N, C, D]``."""
        hidden = F.gelu(self.encoder(series))
        return self.mean_head(hidden), self.logvar_head(hidden)

    def push_forward(self, latent: torch.Tensor, series: torch.Tensor) -> torch.Tensor:
        """Eq. (5): ``latent [N, C, K, D]`` and history ``[N, C, L]`` -> ``[N, C, K, H]``."""
        condition = self.history(series).unsqueeze(-2).expand_as(latent)
        return self.mapping(torch.cat([latent, condition], dim=-1))

    def sample(self, x_enc: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """Forecast samples ``[B, K, pred_len, C]`` in the input scale."""
        count = num_samples or self.num_samples
        mean = x_enc.mean(dim=1, keepdim=True)
        scale = x_enc.std(dim=1, keepdim=True) + self.eps  # unbiased window std, as in the code
        series = ((x_enc - mean) / scale).transpose(1, 2)  # [B, C, L]
        mu, logvar = self.prior(series)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn(*mu.shape[:2], count, mu.shape[-1], device=mu.device, dtype=mu.dtype)
        latent = mu.unsqueeze(-2) + std.unsqueeze(-2) * noise  # Eq. (4)
        draws = self.push_forward(latent, series).permute(0, 2, 3, 1)  # [B, K, H, C]
        return draws * scale.unsqueeze(1) + mean.unsqueeze(1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return empirical_quantiles(self.sample(x_enc), self.quantile_levels, dim=1)

    def training_loss(self, samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Eq. (10): ``alpha * L_NLL + L_MM`` on aligned samples ``[B, K, L, C]``."""
        nll = kde_nll(samples, target, self.bandwidth, self.log_floor)
        return self.alpha * nll + mean_matching(samples, target)

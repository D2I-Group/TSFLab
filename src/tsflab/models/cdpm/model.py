"""CDPM: conditional denoising seasonal module + polynomial trend module (local implementation).

Independent rewrite from Section 4 of the paper (Eqs. 1-16) after reading the pinned
official repository (no license file). The input window fixes instance-normalisation
statistics; both windows are split by an edge-padded moving average into seasonal and
trend parts; a DDIM-sampled conditional diffusion model forecasts the seasonal part and
a two-path (identity / signed square root) linear module forecasts the trend.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.series_decomposition import SeriesDecomposition


def beta_schedule(kind: str, steps: int) -> torch.Tensor:
    """Noise schedule ``beta_1..beta_K`` (cosine schedule clipped at 0.999 by default)."""
    if kind == "cosine":
        s = 0.008
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((grid / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(0, 0.999).float()
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, steps)
    if kind == "exponential":
        return 1e-4 * (0.02 / 1e-4) ** torch.linspace(0, 1, steps)
    return (1e-4 / torch.arange(1, steps + 1).float().sqrt()).clamp(0, 0.999)  # inverse_sqrt


def signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Eq. (13) root path on signed values: ``sign(x) * sqrt(|x|)`` with a finite gradient at 0."""
    return torch.sign(x) * torch.sqrt(x.abs().clamp_min(1e-12))


class StepEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion step ``k`` (``[sin, cos]`` halves)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        self.register_buffer(
            "freqs", torch.exp(-math.log(10000) / (half - 1) * torch.arange(half).float()), persistent=False
        )

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        angle = step.float()[:, None] * self.freqs[None, :]
        return torch.cat((angle.sin(), angle.cos()), dim=-1)


class AdaLayerNorm(nn.Module):
    """Eq. (5): ``(1 + scale_k) * LayerNorm(h) + shift_k`` from the step embedding."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.step = StepEmbedding(emb_dim)
        self.proj = nn.Linear(emb_dim, 2 * emb_dim)
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)

    def forward(self, h: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(F.silu(self.step(step)))[:, None, :].chunk(2, dim=-1)
        return self.norm(h) * (1 + scale) + shift


class DenoiseBlock(nn.Module):
    """AdaLN, a temporal Linear-ReLU mixer with residual, then a feature MLP with residual and LayerNorm."""

    def __init__(self, pred_len: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.ada_norm = AdaLayerNorm(emb_dim)
        self.time_mix = nn.Linear(pred_len, pred_len)
        self.fc1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.fc2 = nn.Linear(4 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, h: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        h = self.ada_norm(h, step)
        mixed = self.dropout(F.relu(self.time_mix(h.transpose(1, 2)))).transpose(1, 2)
        h = mixed + h
        h = self.fc2(self.dropout(F.silu(self.fc1(h)))) + h
        return self.out_norm(self.norm(h))


class ConditionalDenoiser(nn.Module):
    """CDSM network ``X_theta(X^k_{T,s}, k | prior)`` predicting the clean seasonal future (Eqs. 4-11)."""

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        mlp_hidden: int,
        emb_dim: int,
        patch_len: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.in_patches = seq_len // patch_len
        self.out_patches = pred_len // patch_len
        # Eq. (4): convolutional embedding of the noisy seasonal future.
        self.embed = nn.Conv1d(enc_in, emb_dim, kernel_size=3, padding=1)
        self.embed_dropout = nn.Dropout(dropout)
        self.position = nn.Parameter(torch.empty(1, pred_len, emb_dim).uniform_(-0.02, 0.02))
        self.position_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(DenoiseBlock(pred_len, emb_dim, dropout) for _ in range(n_layers))
        self.head = nn.Linear(emb_dim, enc_in)
        # Eqs. (8)-(9): patch statistics of the history mapped to the horizon patches.
        self.mean_mlp = nn.Sequential(
            nn.Linear(self.in_patches, mlp_hidden), nn.LeakyReLU(), nn.Linear(mlp_hidden, self.out_patches)
        )
        self.std_mlp = nn.Sequential(
            nn.Linear(self.in_patches, mlp_hidden), nn.LeakyReLU(), nn.Linear(mlp_hidden, self.out_patches)
        )
        # Eq. (11) with rho_1 = w and rho_2 = 1 - w (official parameterisation, w ~ N(0, 1)).
        self.weight = nn.Parameter(torch.randn(1))

    def patch_statistics(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (6)-(7): per-patch mean and (biased) standard deviation ``[B, L // P, C]``."""
        batch, _, channels = history.shape
        patches = history[:, : self.in_patches * self.patch_len].reshape(
            batch, self.in_patches, self.patch_len, channels
        )
        return patches.mean(dim=2), patches.var(dim=2, unbiased=False).sqrt()

    def prior(self, history: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Eq. (10): Gaussian draw from the predicted horizon-patch mean and scale, ``[B, H, C]``."""
        mean, std = self.patch_statistics(history)
        mean = self.mean_mlp(mean.transpose(1, 2)).transpose(1, 2)
        std = self.std_mlp(std.transpose(1, 2)).transpose(1, 2)
        z = torch.randn(
            mean.shape[0], self.out_patches, self.patch_len, mean.shape[2],
            device=mean.device, dtype=mean.dtype, generator=generator,
        )
        draw = mean[:, :, None, :] + z * std[:, :, None, :]
        return draw.reshape(mean.shape[0], self.pred_len, mean.shape[2])

    def denoise(self, noisy: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        h = self.embed_dropout(self.embed(noisy.transpose(1, 2)).transpose(1, 2))
        h = self.position_dropout(h + self.position)
        for block in self.blocks:
            h = block(h, step)
        return self.head(h)

    def forward(self, noisy, step, history, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.weight * self.denoise(noisy, step) + (1 - self.weight) * self.prior(history, generator)


class PolynomialTrend(nn.Module):
    """PTM (Eqs. 12-14): ``lambda_1 Linear_origin(x) + lambda_2 Linear_root(sqrt(x)) + c``."""

    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.origin = nn.Linear(seq_len, pred_len)
        self.root = nn.Linear(seq_len, pred_len)
        self.mix = nn.Linear(2, 1)

    def forward(self, trend: torch.Tensor) -> torch.Tensor:
        origin = self.origin(trend.transpose(1, 2))
        root = self.root(signed_sqrt(trend).transpose(1, 2))
        return self.mix(torch.stack((origin, root), dim=-1)).squeeze(-1).transpose(1, 2)


class Model(nn.Module):
    """CDPM forecaster: ``forward`` runs DDIM sampling; ``teacher_forced_forecast`` is the training pass."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        diffusion_steps: int = 50,
        schedule: str = "cosine",
        mlp_hidden: int = 256,
        emb_dim: int = 256,
        patch_len: int = 8,
        n_layers: int = 14,
        dropout: float = 0.1,
        moving_avg: int = 5,
        clip_denoised: bool = True,
        eval_seed: int | None = 0,
    ) -> None:
        super().__init__()
        if pred_len % patch_len or seq_len < patch_len:
            raise ValueError("pred_len must be a multiple of patch_len and seq_len >= patch_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.diffusion_steps = diffusion_steps
        self.clip_denoised = clip_denoised
        self.eval_seed = eval_seed
        self.decomposition = SeriesDecomposition(moving_avg)
        self.trend = PolynomialTrend(seq_len, pred_len)
        self.denoiser = ConditionalDenoiser(
            enc_in, seq_len, pred_len, mlp_hidden, emb_dim, patch_len, n_layers, dropout
        )
        betas = beta_schedule(schedule, diffusion_steps)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0), persistent=False)

    @staticmethod
    def statistics(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Instance statistics of the history (mean, biased std with ``1e-5`` inside the root)."""
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        return mean, std

    def q_sample(self, clean: torch.Tensor, step: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Eq. (3): ``sqrt(abar_k) x0 + sqrt(1 - abar_k) eps``."""
        abar = self.alpha_bar[step][:, None, None]
        return abar.sqrt() * clean + (1 - abar).sqrt() * noise

    def teacher_forced_forecast(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Eq. (15): denoise the noised true seasonal future at a random step, add the trend forecast."""
        mean, std = self.statistics(x)
        season_hist, trend_hist = self.decomposition((x - mean) / std)
        season_future, _ = self.decomposition((future - mean) / std)
        step = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device)
        noisy = self.q_sample(season_future, step, torch.randn_like(season_future))
        prediction = self.trend(trend_hist) + self.denoiser(noisy, step, season_hist)
        return prediction * std + mean

    @torch.no_grad()
    def sample_seasonal(self, season_hist: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Deterministic DDIM (eta = 0) over every step ``K-1 .. 0``, ``x0`` clipped to ``[-1, 1]``."""
        batch = season_hist.shape[0]
        sample = torch.randn(
            batch, self.pred_len, self.enc_in, device=season_hist.device, dtype=season_hist.dtype,
            generator=generator,
        )
        for k in range(self.diffusion_steps - 1, -1, -1):
            step = torch.full((batch,), k, device=sample.device, dtype=torch.long)
            clean = self.denoiser(sample, step, season_hist, generator)
            if self.clip_denoised:
                clean = clean.clamp(-1.0, 1.0)
            if k == 0:
                return clean
            abar, abar_next = self.alpha_bar[k], self.alpha_bar[k - 1]
            noise = (sample / abar.sqrt() - clean) / (1 / abar - 1).sqrt()
            sample = clean * abar_next.sqrt() + (1 - abar_next).sqrt() * noise
        return sample

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected x_enc [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        mean, std = self.statistics(x_enc)
        season_hist, trend_hist = self.decomposition((x_enc - mean) / std)
        generator = None
        if not self.training and self.eval_seed is not None:
            generator = torch.Generator(device=x_enc.device).manual_seed(self.eval_seed)
        prediction = self.trend(trend_hist) + self.sample_seasonal(season_hist, generator)
        return prediction * std + mean


__all__ = [
    "AdaLayerNorm",
    "ConditionalDenoiser",
    "Model",
    "PolynomialTrend",
    "beta_schedule",
    "signed_sqrt",
]

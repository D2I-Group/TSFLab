"""Local CCDM implementation from the paper (arXiv 2410.02168) and the pinned official code.

CCDM is a conditional DDPM forecaster for multivariate series. Its denoiser is
channel-aware: channel-independent dense encoders (TiDE-style residual MLPs shared
across channels) embed the history ``x`` and the noisy future ``y_k`` of every
channel, the two embeddings are concatenated per channel, and diffusion-transformer
(DiT) blocks with adaptive layer norm on the diffusion-step embedding mix the
channel tokens with self-attention. Training adds a denoising-based InfoNCE term
(Eqs. 3-4) whose negatives are patch-shuffled and amplitude-scaled copies of the
target; inference draws DDPM ancestral samples and returns their quantiles.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.empirical_quantiles import empirical_quantiles


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------
def beta_schedule(kind: str, steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Per-step noise levels ``beta_1..beta_K`` (float64).

    ``quad`` (the paper's "quadratic" schedule, Table 5) interpolates
    ``sqrt(beta)`` linearly; ``linear`` interpolates ``beta``; ``cosine`` is the
    improved-DDPM schedule with offset ``s = 0.008`` clipped at 0.999 (it ignores
    ``beta_start`` / ``beta_end``).
    """
    if kind == "quad":
        return torch.linspace(beta_start**0.5, beta_end**0.5, steps, dtype=torch.float64) ** 2
    if kind == "linear":
        return torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    if kind == "cosine":
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((grid / steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, 0.999)
    raise ValueError(f"unknown beta schedule {kind!r}")


# ---------------------------------------------------------------------------
# Denoiser building blocks
# ---------------------------------------------------------------------------
class DenseResidual(nn.Module):
    """CiDM residual block: ``LayerNorm(Dropout(W2 ReLU(W1 x)) + W_res x)`` (TiDE)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_dim, out_dim)
        self.output = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense = self.dropout(self.output(F.relu(self.hidden(x))))
        return self.norm(dense + self.skip(x))


class DenseEncoder(nn.Module):
    """``layers`` stacked CiDMs over the last axis; applied to ``[B, D, length]`` it is
    channel-independent (one set of weights shared by every channel)."""

    def __init__(self, in_dim: int, out_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("a dense encoder needs at least one CiDM")
        widths = [in_dim] + [out_dim] * layers
        self.blocks = nn.Sequential(
            *(DenseResidual(a, b, dropout) for a, b in zip(widths[:-1], widths[1:]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class StepEmbedding(nn.Module):
    """Diffusion-step embedding: ``[cos(k f), sin(k f)]`` with ``f_i = 10000^(-i/half)``,
    then ``Linear -> SiLU -> Linear``."""

    def __init__(self, hidden: int, frequency_dim: int = 256, max_period: float = 10000.0) -> None:
        super().__init__()
        if frequency_dim % 2:
            raise ValueError("frequency_dim must be even")
        half = frequency_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("frequencies", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def sinusoid(self, k: torch.Tensor) -> torch.Tensor:
        angles = k.float()[:, None] * self.frequencies[None]
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoid(k))


class ChannelAttention(nn.Module):
    """Multi-head self-attention over the channel-token axis of ``[B, D, d_model]``.

    Bias-free query/key/value projections, softmax(QK^T / sqrt(d_head)) with dropout
    on the attention probabilities, biased output projection.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        heads = self.n_heads

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, tokens, heads, width // heads).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // heads)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, width)
        return self.out(mixed)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN: ``x * (1 + scale) + shift`` with per-sample ``shift``/``scale``."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ChannelDiTBlock(nn.Module):
    """DiT block over channel tokens, conditioned on the step embedding ``c``.

    ``h + g1 * Attn(adaLN(LN(h)))`` then ``h + g2 * MLP(adaLN(LN(h)))``; the six
    shift/scale/gate vectors come from ``Linear(SiLU(c))``.
    """

    def __init__(self, d_model: int, cond_dim: int, n_heads: int, attn_dropout: float,
                 mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = ChannelAttention(d_model, n_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = max(1, int(d_model * mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, d_model))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * d_model))

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, dim=1)
        h = h + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(h), shift1, scale1))
        return h + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(h), shift2, scale2))


class OutputDecoder(nn.Module):
    """Final adaLN (shift/scale from ``c``), ``layers`` CiDMs and a linear map to the horizon."""

    def __init__(self, d_model: int, cond_dim: int, pred_len: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * d_model))
        self.dense = DenseEncoder(d_model, d_model, layers, dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=1)
        return self.head(self.dense(modulate(self.norm(h), shift, scale)))


class ChannelAwareDenoiser(nn.Module):
    """``eps_theta(y_k, x, k)``: ``[B, H, D]`` noise estimate (Sec. 3.1, Fig. 3)."""

    def __init__(self, seq_len: int, pred_len: int, cont_hidden_dim: int, pred_hidden_dim: int,
                 step_hidden_dim: int, encoder_layers: int, decoder_layers: int, n_depth: int,
                 n_heads: int, attn_dropout: float, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        d_model = cont_hidden_dim + pred_hidden_dim
        self.condition_encoder = DenseEncoder(seq_len, cont_hidden_dim, encoder_layers, dropout)
        self.latent_encoder = DenseEncoder(pred_len, pred_hidden_dim, encoder_layers, dropout)
        self.step_embedding = StepEmbedding(step_hidden_dim)
        self.blocks = nn.ModuleList(
            ChannelDiTBlock(d_model, step_hidden_dim, n_heads, attn_dropout, mlp_ratio)
            for _ in range(n_depth)
        )
        self.decoder = OutputDecoder(d_model, step_hidden_dim, pred_len, decoder_layers, dropout)

    def forward(self, x: torch.Tensor, y_k: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Channel-independent encoders on [B, D, time]; channel tokens hold [x-code ; y-code].
        h = torch.cat([self.condition_encoder(x.transpose(1, 2)),
                       self.latent_encoder(y_k.transpose(1, 2))], dim=-1)
        c = self.step_embedding(k)
        for block in self.blocks:
            h = block(h, c)  # channel mixing
        return self.decoder(h, c).transpose(1, 2)


# ---------------------------------------------------------------------------
# Negative construction (Appendix A.3)
# ---------------------------------------------------------------------------
def patch_shuffle(y0: torch.Tensor, count: int, patch_len: int) -> torch.Tensor:
    """``count`` negatives per target by permuting length-``patch_len`` patches.

    Returns ``[B, count, H, D]``. Negative ``n`` uses one random patch order for the
    whole batch and all channels; a trailing remainder shorter than a patch stays in
    place.
    """
    batch, horizon, channels = y0.shape
    patches = horizon // patch_len
    covered = patches * patch_len
    if patches < 2:
        return y0.unsqueeze(1).expand(batch, count, horizon, channels).clone()
    blocks = y0[:, :covered].reshape(batch, patches, patch_len, channels)
    order = torch.stack([torch.randperm(patches, device=y0.device) for _ in range(count)])
    shuffled = blocks[:, order].reshape(batch, count, covered, channels)
    tail = y0[:, covered:].unsqueeze(1).expand(batch, count, horizon - covered, channels)
    return torch.cat([shuffled, tail], dim=2)


def amplitude_scale(y0: torch.Tensor, count: int) -> torch.Tensor:
    """``count`` negatives ``a_d * y0_d``: half with ``a_d ~ U[1.5, 2]``, the rest with
    ``a_d ~ U[0, 0.5]``, drawn per target and channel. Returns ``[B, count, H, D]``."""
    batch, _, channels = y0.shape
    up = count // 2
    factors = torch.cat([
        torch.empty(batch, up, 1, channels, device=y0.device, dtype=y0.dtype).uniform_(1.5, 2.0),
        torch.empty(batch, count - up, 1, channels, device=y0.device, dtype=y0.dtype).uniform_(0.0, 0.5),
    ], dim=1)
    return y0.unsqueeze(1) * factors


def _check_contrast(contrast_form: str, contrast_weight: float, n_negatives: int, temperature: float) -> None:
    if contrast_form not in {"regression", "similarity"}:
        raise ValueError("contrast_form must be 'regression' or 'similarity'")
    if contrast_weight < 0 or n_negatives < 0 or temperature <= 0:
        raise ValueError("invalid contrastive settings")


def _checked_levels(quantile_levels: list[float]) -> list[float]:
    levels = list(quantile_levels)
    if not levels or any(not 0.0 < q < 1.0 for q in levels) or levels != sorted(set(levels)):
        raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
    return levels


class Model(nn.Module):
    """CCDM: channel-aware conditional DDPM with denoising-based contrastive refinement.

    ``forward`` returns empirical quantiles ``[B, pred_len, enc_in, Q]`` of
    ``num_samples`` ancestral DDPM trajectories; ``training_loss`` is Eq. 6.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        quantile_levels: list[float],
        cont_hidden_dim: int = 128,
        pred_hidden_dim: int = 128,
        step_hidden_dim: int = 128,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        n_depth: int = 2,
        n_heads: int = 8,
        attn_dropout: float = 0.1,
        mlp_ratio: float = 1.0,
        dropout: float = 0.1,
        diffusion_steps: int = 50,
        beta_schedule_kind: str = "quad",
        beta_start: float = 1e-4,
        beta_end: float = 0.5,
        window_norm: bool = True,
        contrast_weight: float = 1e-3,
        n_negatives: int = 64,
        temperature: float = 0.1,
        shuffle_patch_len: int = 8,
        contrast_form: str = "regression",
        num_samples: int = 100,
        sample_batch_size: int = 100,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, diffusion_steps, num_samples, sample_batch_size,
               shuffle_patch_len) < 1:
            raise ValueError("CCDM sizes must be positive")
        if (cont_hidden_dim + pred_hidden_dim) % n_heads:
            raise ValueError("cont_hidden_dim + pred_hidden_dim must be divisible by n_heads")
        _check_contrast(contrast_form, contrast_weight, n_negatives, temperature)
        levels = _checked_levels(quantile_levels)
        self.output_type = "quantile"
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.num_steps = diffusion_steps
        self.window_norm = window_norm
        self.contrast_weight = contrast_weight
        self.n_negatives = n_negatives
        self.temperature = temperature
        self.shuffle_patch_len = shuffle_patch_len
        self.contrast_form = contrast_form
        self.num_samples = num_samples
        self.sample_batch_size = sample_batch_size
        self.register_buffer("quantile_levels", torch.tensor(levels), persistent=False)

        betas = beta_schedule(beta_schedule_kind, diffusion_steps, beta_start, beta_end)
        if not ((betas > 0) & (betas < 1)).all():
            raise ValueError("noise levels must lie in (0, 1)")
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])
        schedule = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": alpha_bar,
            "alpha_bar_prev": alpha_bar_prev,
            "posterior_variance": betas * (1 - alpha_bar_prev) / (1 - alpha_bar),
        }
        for name, value in schedule.items():
            self.register_buffer(name, value.float(), persistent=False)

        self.denoiser = ChannelAwareDenoiser(
            seq_len, pred_len, cont_hidden_dim, pred_hidden_dim, step_hidden_dim,
            encoder_layers, decoder_layers, n_depth, n_heads, attn_dropout, mlp_ratio, dropout,
        )

    # -- normalization -------------------------------------------------------
    def window_statistics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """History mean and ``sqrt(var + 1e-5)`` per sample and channel, ``[B, 1, D]``."""
        if not self.window_norm:
            ones = torch.ones_like(x[:, :1])
            return torch.zeros_like(ones), ones
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        return mean, std

    # -- forward diffusion ---------------------------------------------------
    def _at(self, name: str, k: torch.Tensor) -> torch.Tensor:
        return getattr(self, name)[k].view(-1, 1, 1)

    def q_sample(self, y0: torch.Tensor, k: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """``y_k = sqrt(abar_k) y0 + sqrt(1 - abar_k) eps``."""
        abar = self._at("alpha_bar", k)
        return abar.sqrt() * y0 + (1 - abar).sqrt() * noise

    def sample_steps(self, batch: int, device) -> torch.Tensor:
        """Antithetic uniform steps ``k`` and ``K - 1 - k`` (official training loop)."""
        half = torch.randint(0, self.num_steps, (batch // 2 + 1,), device=device)
        return torch.cat([half, self.num_steps - 1 - half])[:batch]

    # -- reverse diffusion ---------------------------------------------------
    def reverse_step(self, x: torch.Tensor, y_k: torch.Tensor, step: int) -> torch.Tensor:
        """One ancestral step ``p(y_{k-1} | y_k, x)`` with the posterior variance."""
        k = torch.full((y_k.shape[0],), step, device=y_k.device, dtype=torch.long)
        eps = self.denoiser(x, y_k, k)
        mean = (y_k - self._at("betas", k) / (1 - self._at("alpha_bar", k)).sqrt() * eps) \
            / self._at("alphas", k).sqrt()
        variance = self._at("posterior_variance", k)
        return mean + variance.sqrt() * torch.randn_like(y_k)

    def _sample_chunk(self, x: torch.Tensor, count: int) -> torch.Tensor:
        """``[count, B, pred_len, D]`` ancestral trajectories from Gaussian noise, normalized scale."""
        batch = x.shape[0]
        condition = x.repeat(count, 1, 1)
        y = torch.randn(count * batch, self.pred_len, self.enc_in, device=x.device, dtype=x.dtype)
        for step in reversed(range(self.num_steps)):
            y = self.reverse_step(condition, y, step)
        return y.view(count, batch, self.pred_len, self.enc_in)

    @torch.no_grad()
    def sample(self, x_enc: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """``[S, B, pred_len, D]`` forecasts in the input scale."""
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(f"CCDM expects [batch, {self.seq_len}, {self.enc_in}]")
        mean, std = self.window_statistics(x_enc)
        x = (x_enc - mean) / std
        chunks, remaining = [], num_samples or self.num_samples
        while remaining > 0:
            count = min(remaining, self.sample_batch_size)
            chunks.append(self._sample_chunk(x, count))
            remaining -= count
        samples = torch.cat(chunks, dim=0)
        return samples * std.unsqueeze(0) + mean.unsqueeze(0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        samples = self.sample(x_enc)
        return empirical_quantiles(samples, self.quantile_levels.to(samples.dtype))

    # -- training (Eqs. 1, 3, 4, 6) -----------------------------------------
    def negatives(self, y0: torch.Tensor) -> torch.Tensor:
        """``[B, 2N, H, D]``: N patch-shuffled then N amplitude-scaled targets."""
        return torch.cat([patch_shuffle(y0, self.n_negatives, self.shuffle_patch_len),
                          amplitude_scale(y0, self.n_negatives)], dim=1)

    def contrastive_loss(self, x: torch.Tensor, y0: torch.Tensor, k: torch.Tensor,
                         negatives: torch.Tensor) -> torch.Tensor:
        """InfoNCE of Eq. 4 with the positive in slot 0.

        All candidates (positive and negatives) share step ``k`` and one fresh noise
        draw ``eps'``. ``regression``: score ``-mean((eps' - eps_theta)^2) / tau``
        (Eq. 3); ``similarity``: ``cos(eps_theta, eps') / tau`` over the flattened
        window (the official training call).
        """
        batch, count = negatives.shape[:2]
        candidates = torch.cat([y0.unsqueeze(1), negatives], dim=1)  # [B, 1 + 2N, H, D]
        flat = candidates.reshape(batch * (count + 1), self.pred_len, self.enc_in)
        noise = torch.randn_like(y0).repeat_interleave(count + 1, dim=0)
        steps = k.repeat_interleave(count + 1)
        condition = x.repeat_interleave(count + 1, dim=0)
        predicted = self.denoiser(condition, self.q_sample(flat, steps, noise), steps)
        if self.contrast_form == "regression":
            scores = -(noise - predicted).square().flatten(1).mean(dim=1)
        else:
            scores = F.cosine_similarity(predicted.flatten(1), noise.flatten(1), dim=1)
        logits = scores.view(batch, count + 1) / self.temperature
        return -torch.log_softmax(logits, dim=1)[:, 0].mean()

    def training_loss(self, x_enc: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """``L_denoise + lambda * L_contrast`` at one random step per sample (Eq. 6)."""
        mean, std = self.window_statistics(x_enc)
        x, y0 = (x_enc - mean) / std, (future - mean) / std
        k = self.sample_steps(x.shape[0], x.device)
        noise = torch.randn_like(y0)
        denoise = (noise - self.denoiser(x, self.q_sample(y0, k, noise), k)).square().mean()
        if self.contrast_weight == 0 or self.n_negatives == 0:
            return denoise
        contrast = self.contrastive_loss(x, y0, k, self.negatives(y0))
        return denoise + self.contrast_weight * contrast

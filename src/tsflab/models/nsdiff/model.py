"""NsDiff: non-stationary diffusion for probabilistic forecasting (local rewrite).

The future window follows a location-scale noise model ``Y = f(X) + sqrt(g(X)) eps``
(paper Eq. 1). A Non-stationary-Transformer prior ``f`` gives the conditional mean,
a sliding-variance MLP ``g`` gives the time-varying variance, and a conditional
denoiser ``xi`` estimates both the forward noise and the reverse-step variance. The
forward chain moves the variance from the data variance ``sigma_Y0`` toward the
endpoint ``g(X)`` (uncertainty-aware schedule, Eq. 6-8); inference starts from
``N(f(X), g(X))`` and recovers ``sigma_Y0`` from the denoiser's variance by solving
the quadratic of Eq. 15-18 at every reverse step.

Independent implementation from the paper and the official repository (no license)
at the revision pinned in the model card; no official source was copied.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding
from tsflab.models._components.empirical_quantiles import empirical_quantiles
from tsflab.models._components.masking import TriangularCausalMask
from tsflab.models._components.self_attention_family import AttentionLayer
from tsflab.models._components.transformer_encdec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)
from tsflab.models._inputs import check_window

# Numerical floor added to variances during training (official constant ``10e-8``).
VARIANCE_FLOOR = 1e-7


# ---------------------------------------------------------------------------
# Uncertainty-aware noise schedule (Eq. 6-8)
# ---------------------------------------------------------------------------


def uncertainty_schedule(steps: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    """Return the linear DDPM schedule and the NsDiff coefficients, 0-based in ``t``.

    ``alpha_tilde_t = sum_k prod_{i=t-k}^{t} alpha_i`` and
    ``alpha_hat_t = sum_k alpha_{t-k} prod_{i=t-k}^{t} alpha_i`` (Eq. 8) satisfy the
    recurrences ``alpha_tilde_t = alpha_t (1 + alpha_tilde_{t-1})`` and
    ``alpha_hat_t = alpha_t (alpha_t + alpha_hat_{t-1})``. ``beta_bar = 1 - alpha_bar``
    and ``beta_tilde = alpha_tilde - alpha_hat``. The ``*_prev`` entries hold the
    value at ``t - 1`` with the official convention of ``1`` at ``t = 0``.
    """
    betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_tilde = torch.empty_like(alphas)
    alpha_hat = torch.empty_like(alphas)
    running_tilde = running_hat = 0.0
    for index, alpha in enumerate(alphas):
        running_tilde = alpha * (1.0 + running_tilde)
        running_hat = alpha * (alpha + running_hat)
        alpha_tilde[index] = running_tilde
        alpha_hat[index] = running_hat
    beta_bar = 1.0 - alpha_bar
    beta_tilde = alpha_tilde - alpha_hat
    if (beta_tilde < 0).any() or (beta_bar - beta_tilde < -1e-12).any():
        raise ValueError("noise schedule violates 0 <= beta_tilde <= beta_bar")
    one = torch.ones(1, dtype=torch.float64)
    table = {
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": torch.cat([one, alpha_bar[:-1]]),
        "beta_bar": beta_bar,
        "beta_tilde": beta_tilde,
        "beta_bar_prev": torch.cat([one, beta_bar[:-1]]),
        "beta_tilde_prev": torch.cat([one, beta_tilde[:-1]]),
    }
    return {name: value.float() for name, value in table.items()}


def trailing_variance(series: torch.Tensor, window: int) -> torch.Tensor:
    """Population variance of each length-``window`` window ending at every step.

    ``series`` is ``[B, T, C]``; the result is ``[B, T - window + 1, C]``.
    """
    return series.unfold(1, window, 1).var(dim=-1, unbiased=False)


def target_variance(history: torch.Tensor, future: torch.Tensor, window: int) -> torch.Tensor:
    """Ground-truth variance ``sigma_Y0`` of Eq. 14 for every future step.

    Variance over the trailing window of the joined history and future, with the
    first observation replicated in front so every step has a full window.
    """
    joined = torch.cat([history, future], dim=1)
    padded = torch.cat([joined[:, :1].expand(-1, window, -1), joined], dim=1)
    return trailing_variance(padded, window)[:, -future.shape[1] :, :]


# ---------------------------------------------------------------------------
# Prior mean f_phi: Non-stationary Transformer
# ---------------------------------------------------------------------------


class DeStationaryAttention(nn.Module):
    """Attention core ``softmax(scale (tau Q K^T + delta))`` for ``AttentionLayer``.

    ``tau`` is ``[B, 1]`` (positive) and ``delta`` ``[B, S]`` over the key axis; either
    may be ``None`` (identity). Inputs are ``[B, L, H, E]``; output ``[B, L, H, D]``.
    """

    def __init__(self, causal: bool, dropout: float) -> None:
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch, length, _, width = queries.shape
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if tau is not None:
            scores = scores * tau.view(batch, 1, 1, 1)
        if delta is not None:
            scores = scores + delta.view(batch, 1, 1, -1)
        if self.causal:
            mask = attn_mask if attn_mask is not None else TriangularCausalMask(batch, length, device=queries.device)
            scores = scores.masked_fill(mask.mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores / sqrt(width), dim=-1))
        return torch.einsum("bhls,bshd->blhd", weights, values).contiguous(), None


class DeStationaryFactorProjector(nn.Module):
    """MLP that learns ``tau`` or ``delta`` from the raw series and one removed statistic.

    The time axis is mixed by a bias-free circular convolution over channels (time
    steps act as input channels), concatenated with the statistic, then an MLP.
    """

    def __init__(self, seq_len: int, channels: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        self.time_mixer = nn.Conv1d(seq_len, 1, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        widths = [2 * channels, *hidden_dims]
        layers: list[nn.Module] = []
        for left, right in zip(widths, widths[1:]):
            layers += [nn.Linear(left, right), nn.ReLU()]
        layers.append(nn.Linear(widths[-1], output_dim, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, raw: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
        mixed = self.time_mixer(raw)  # [B, 1, C]
        return self.network(torch.cat([mixed, statistic], dim=1).flatten(1))


class StationaryPriorMean(nn.Module):
    """Non-stationary Transformer predicting ``f_phi(X) = E[Y | X]``.

    Series stationarization (per-window mean/std), de-stationary attention in an
    encoder-decoder whose decoder sees the last ``label_len`` normalized steps
    followed by zeros, and de-normalization of the projected output.
    """

    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        dropout: float,
        activation: str,
        p_hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.label_len, self.pred_len = label_len, pred_len
        self.encoder_embedding = DataEmbedding(enc_in, d_model, "timeF", "h", dropout)
        self.decoder_embedding = DataEmbedding(enc_in, d_model, "timeF", "h", dropout)

        def attention(causal: bool) -> AttentionLayer:
            return AttentionLayer(DeStationaryAttention(causal, dropout), d_model, n_heads)

        self.encoder = Encoder(
            [EncoderLayer(attention(False), d_model, d_ff, dropout, activation) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.decoder = Decoder(
            [
                DecoderLayer(attention(True), attention(False), d_model, d_ff, dropout, activation)
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, enc_in),
        )
        self.tau_learner = DeStationaryFactorProjector(seq_len, enc_in, p_hidden_dims, 1)
        self.delta_learner = DeStationaryFactorProjector(seq_len, enc_in, p_hidden_dims, seq_len)

    def forward(self, x, x_mark=None, y_mark=None) -> torch.Tensor:
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = (x - mean) / std
        tau = self.tau_learner(x, std).exp()  # [B, 1]
        delta = self.delta_learner(x, mean)  # [B, seq_len]
        decoder_values = torch.cat(
            [normalized[:, x.shape[1] - self.label_len :], normalized.new_zeros(x.shape[0], self.pred_len, x.shape[2])],
            dim=1,
        )
        if y_mark is not None:
            y_mark = y_mark[:, -(self.label_len + self.pred_len) :]
        memory, _ = self.encoder(self.encoder_embedding(normalized, x_mark), tau=tau, delta=delta)
        decoded = self.decoder(self.decoder_embedding(decoder_values, y_mark), memory, tau=tau, delta=delta)
        return decoded[:, -self.pred_len :] * std + mean


# ---------------------------------------------------------------------------
# Prior variance g_psi and the denoiser xi_theta
# ---------------------------------------------------------------------------


class SlidingVarianceEstimator(nn.Module):
    """``g_psi(X)``: three-layer MLP from input sliding variances to future variances.

    Trailing-window variances of the history (dropping the first window, as in the
    official code) are mapped over time per channel and passed through softplus.
    """

    def __init__(self, seq_len: int, pred_len: int, window: int, hidden: int) -> None:
        super().__init__()
        self.window = window
        self.network = nn.Sequential(
            nn.Linear(seq_len - window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variances = trailing_variance(x, self.window)[:, 1:] + VARIANCE_FLOOR
        return F.softplus(self.network(variances.transpose(1, 2))).transpose(1, 2)


class StepScaledLinear(nn.Module):
    """Linear map whose output is scaled elementwise by a learned per-step vector."""

    def __init__(self, in_features: int, out_features: int, steps: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.step_scale = nn.Embedding(steps, out_features)
        nn.init.uniform_(self.step_scale.weight, 0.0, 1.0)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.linear(h) * self.step_scale(t).unsqueeze(1)


class NoiseVarianceDenoiser(nn.Module):
    """``xi_theta(Y_t, f(X), g(X), t) -> (eta_theta, sigma_theta)`` per time step.

    The three conditioning series are concatenated on the channel axis, passed through
    three step-scaled softplus layers; a linear head predicts the noise and a softplus
    head on the softplus of the last hidden state predicts the reverse variance.
    """

    def __init__(self, channels: int, hidden: int, steps: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StepScaledLinear(3 * channels, hidden, steps + 1),
                StepScaledLinear(hidden, hidden, steps + 1),
                StepScaledLinear(hidden, hidden, steps + 1),
            ]
        )
        self.noise_head = nn.Linear(hidden, channels)
        self.variance_head = nn.Linear(hidden, channels)

    def forward(self, y_t, prior_mean, prior_variance, t):
        h = torch.cat([y_t, prior_mean, prior_variance], dim=-1)
        for layer in self.layers:
            h = F.softplus(layer(h, t))
        return self.noise_head(h), F.softplus(self.variance_head(F.softplus(h)))


# ---------------------------------------------------------------------------
# NsDiff
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """NsDiff with an end-to-end trained prior mean, prior variance and denoiser.

    ``forward`` draws ``num_samples`` reverse-diffusion trajectories and returns their
    empirical quantiles ``[B, pred_len, enc_in, K]``; ``training_loss`` is the joint
    objective of Eq. 13 plus the prior-mean and prior-variance regressions.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        label_len: int,
        enc_in: int,
        quantile_levels: list[float],
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 1024,
        dropout: float = 0.05,
        activation: str = "gelu",
        p_hidden_dims: list[int] | None = None,
        variance_hidden: int = 512,
        rolling_length: int = 96,
        denoiser_hidden: int = 128,
        diffusion_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 0.01,
        num_samples: int = 100,
        sample_batch_size: int = 10,
    ) -> None:
        super().__init__()
        if rolling_length >= seq_len:
            raise ValueError("rolling_length must satisfy 0 < rolling_length < seq_len")
        if label_len > seq_len:
            raise ValueError("label_len must not exceed seq_len")
        levels = list(quantile_levels)
        if not levels or any(not 0.0 < q < 1.0 for q in levels) or levels != sorted(set(levels)):
            raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
        self.output_type = "quantile"
        self.seq_len, self.pred_len, self.label_len, self.enc_in = seq_len, pred_len, label_len, enc_in
        self.rolling_length = rolling_length
        self.num_steps = diffusion_steps
        self.num_samples = num_samples
        self.sample_batch_size = sample_batch_size
        self.register_buffer("quantile_levels", torch.tensor(levels), persistent=False)
        for name, value in uncertainty_schedule(diffusion_steps, beta_start, beta_end).items():
            self.register_buffer(name, value, persistent=False)

        self.prior_mean = StationaryPriorMean(
            seq_len,
            label_len,
            pred_len,
            enc_in,
            d_model,
            n_heads,
            e_layers,
            d_layers,
            d_ff,
            dropout,
            activation,
            list(p_hidden_dims or [64, 64]),
        )
        self.prior_variance = SlidingVarianceEstimator(seq_len, pred_len, rolling_length, variance_hidden)
        self.denoiser = NoiseVarianceDenoiser(enc_in, denoiser_hidden, diffusion_steps)

    # -- schedule helpers -------------------------------------------------
    def _at(self, name: str, t: torch.Tensor) -> torch.Tensor:
        return getattr(self, name)[t].view(-1, 1, 1)

    def marginal_variance(self, t, prior_variance, data_variance, *, previous: bool = False):
        """Eq. 7: ``(beta_bar_t - beta_tilde_t) g + beta_tilde_t sigma_Y0`` (or at ``t-1``)."""
        suffix = "_prev" if previous else ""
        beta_bar = self._at("beta_bar" + suffix, t)
        beta_tilde = self._at("beta_tilde" + suffix, t)
        return (beta_bar - beta_tilde) * prior_variance + beta_tilde * data_variance

    def step_variance(self, t, prior_variance, data_variance):
        """Eq. 6: ``beta_t^2 g + alpha_t beta_t sigma_Y0``."""
        alpha = self._at("alphas", t)
        beta = 1.0 - alpha
        return beta * beta * prior_variance + alpha * beta * data_variance

    def posterior_variance(self, t, prior_variance, data_variance):
        """Eq. 9: ``sigma_t sigma_bar_{t-1} / (alpha_t sigma_bar_{t-1} + sigma_t)``."""
        alpha = self._at("alphas", t)
        sigma_t = self.step_variance(t, prior_variance, data_variance)
        sigma_prev = self.marginal_variance(t, prior_variance, data_variance, previous=True)
        return sigma_t * sigma_prev / (alpha * sigma_prev + sigma_t)

    def posterior_coefficients(self, t, prior_variance, data_variance):
        """Eq. 10-12: ``gamma_0, gamma_1, gamma_2`` of the posterior mean.

        ``gamma_2`` uses ``sqrt(alpha_t) (alpha_t - 1)`` as printed in Eq. 12 and used by
        the official code (exact conditioning on Eq. 6 would give ``sqrt(alpha_t) - 1``).
        """
        alpha = self._at("alphas", t)
        sqrt_alpha_bar_prev = self._at("alpha_bar_prev", t).sqrt()
        sigma_t = self.step_variance(t, prior_variance, data_variance)
        sigma_prev = self.marginal_variance(t, prior_variance, data_variance, previous=True)
        denominator = alpha * sigma_prev + sigma_t
        gamma0 = sqrt_alpha_bar_prev * sigma_t / denominator
        gamma1 = alpha.sqrt() * sigma_prev / denominator
        gamma2 = (alpha.sqrt() * (alpha - 1.0) * sigma_prev + (1.0 - sqrt_alpha_bar_prev) * sigma_t) / denominator
        return gamma0, gamma1, gamma2

    def estimate_data_variance(self, t, prior_variance, reverse_variance):
        """Eq. 15-18: positive root of ``lambda_0 s^2 + lambda_1 s + lambda_2 = 0``.

        The discriminant is floored at ``1e-20`` and the root at zero; both are the
        identity whenever the solvability condition of Eq. 17 holds.
        """
        alpha = self._at("alphas", t)
        beta = 1.0 - alpha
        bar_prev = self._at("beta_bar_prev", t)
        tilde_prev = self._at("beta_tilde_prev", t)
        gap = bar_prev - tilde_prev
        g, s = prior_variance, reverse_variance
        lambda0 = alpha * beta * tilde_prev
        lambda1 = (beta * beta * tilde_prev + alpha * beta * gap) * g - s * (alpha * tilde_prev + alpha * beta)
        lambda2 = beta * beta * gap * g * g - s * g * (alpha * gap + beta * beta)
        discriminant = (lambda1 * lambda1 - 4.0 * lambda0 * lambda2).clamp_min(1e-20)
        return ((-lambda1 + discriminant.sqrt()) / (2.0 * lambda0)).clamp_min(0.0)

    def diffuse(self, y0, prior_mean, prior_variance, data_variance, t, noise):
        """Closed-form forward sample of Eq. 7."""
        sqrt_alpha_bar = self._at("alpha_bar", t).sqrt()
        std = self.marginal_variance(t, prior_variance, data_variance).sqrt()
        return sqrt_alpha_bar * y0 + (1.0 - sqrt_alpha_bar) * prior_mean + std * noise

    # -- priors ------------------------------------------------------------
    def priors(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        """Return ``f_phi(X)`` and ``g_psi(X)``, both ``[B, pred_len, enc_in]``."""
        check_window(x_enc, self.seq_len, self.enc_in)
        return self.prior_mean(x_enc, x_mark_enc, x_mark_dec), self.prior_variance(x_enc)

    # -- reverse process (Algorithm 2) ---------------------------------------
    def reverse_step(self, y_t, prior_mean, prior_variance, step: int):
        """One reverse transition from ``Y_t`` (0-based ``step``)."""
        t = torch.full((y_t.shape[0],), step, dtype=torch.long, device=y_t.device)
        noise_hat, reverse_variance = self.denoiser(y_t, prior_mean, prior_variance, t)
        data_variance = self.estimate_data_variance(t, prior_variance, reverse_variance)
        sqrt_alpha_bar = self._at("alpha_bar", t).sqrt()
        std = self.marginal_variance(t, prior_variance, data_variance).sqrt()
        y0_hat = (y_t - (1.0 - sqrt_alpha_bar) * prior_mean - std * noise_hat) / sqrt_alpha_bar
        if step == 0:
            return y0_hat
        gamma0, gamma1, gamma2 = self.posterior_coefficients(t, prior_variance, data_variance)
        mean = gamma0 * y0_hat + gamma1 * y_t + gamma2 * prior_mean
        return mean + reverse_variance.sqrt() * torch.randn_like(y_t)

    def sample_from_priors(self, prior_mean, prior_variance, num_samples: int) -> torch.Tensor:
        """Draw ``[num_samples, B, pred_len, enc_in]`` forecasts from the endpoint."""
        chunks = []
        remaining = num_samples
        while remaining > 0:
            count = min(remaining, self.sample_batch_size)
            mean = prior_mean.repeat(count, 1, 1)
            variance = prior_variance.repeat(count, 1, 1)
            y = mean + variance.sqrt() * torch.randn_like(mean)  # Y_T ~ N(f, g)
            for step in reversed(range(self.num_steps)):
                y = self.reverse_step(y, mean, variance, step)
            chunks.append(y.view(count, *prior_mean.shape))
            remaining -= count
        return torch.cat(chunks, dim=0)

    def sample(self, x_enc, x_mark_enc=None, x_mark_dec=None, num_samples: int | None = None):
        prior_mean, prior_variance = self.priors(x_enc, x_mark_enc, x_mark_dec)
        return self.sample_from_priors(prior_mean, prior_variance, num_samples or self.num_samples)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        samples = self.sample(x_enc, x_mark_enc, x_mark_dec)
        return empirical_quantiles(samples, self.quantile_levels.to(samples.dtype))

    # -- training (Algorithm 1, Eq. 13) --------------------------------------
    def training_loss(self, x_enc, x_mark_enc, x_mark_dec, future) -> torch.Tensor:
        """Joint loss: noise and variance KL terms (Eq. 13) + MSE of f + sqrt-variance of g."""
        prior_mean, prior_variance = self.priors(x_enc, x_mark_enc, x_mark_dec)
        prior_variance = prior_variance + VARIANCE_FLOOR
        data_variance = target_variance(x_enc, future, self.rolling_length) + VARIANCE_FLOOR
        batch = x_enc.shape[0]
        # Antithetic timestep draw, as in the official training loop.
        half = torch.randint(0, self.num_steps, (batch // 2 + 1,), device=x_enc.device)
        t = torch.cat([half, self.num_steps - 1 - half])[:batch]
        noise = torch.randn_like(future)
        y_t = self.diffuse(future, prior_mean, prior_variance, data_variance, t, noise)
        noise_hat, reverse_variance = self.denoiser(y_t, prior_mean, prior_variance, t)
        reverse_variance = reverse_variance + VARIANCE_FLOOR
        ratio = self.posterior_variance(t, prior_variance, data_variance) / reverse_variance
        kl = (noise - noise_hat).square().mean() + ratio.mean() - ratio.log().mean()
        mean_loss = (prior_mean - future).square().mean()
        variance_loss = (prior_variance.sqrt() - data_variance.sqrt()).square().mean()
        return kl + mean_loss + variance_loss

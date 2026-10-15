"""StaTS: a learned spectral-trajectory noise schedule with a frequency-guided denoiser.

* Spectral Trajectory Scheduler (STS, Sec. 3.1.1): the variance schedule ``beta``
  is a learnable vector (initialised linearly from ``beta_start`` to ``beta_end``
  and clamped to ``[1e-6, 1 - 1e-6]``, as in the official code), regularised by
  the barrier, spectral-endpoint, initial-step, flatness-progression and
  smoothness objectives plus the forecasting objective (Eqs. 6-13).
* Frequency Guided Denoiser (FGD, Sec. 3.1.2): a conditional guided module
  (energy-gated, multi-band complex-scaled history spectrum projected to the
  horizon, Eqs. 15-20) and a spectral conditioned denoising module (history
  spectral distortion gate, two input projections, two FiLM layers, Eqs. 21-25),
  fused by a learned sigmoid weight (Eq. 26); FGD predicts ``x0`` (Eq. 3).
* Training (Sec. 3.2): ``pretrain`` alternates FGD and STS epochs (stage I), then
  freezes the schedule; the training objective fits FGD (stage II). ``forward``
  draws deterministic DDIM (eta = 0) trajectories from ``x_T ~ N(0, I)`` and
  returns their empirical quantiles.

Independent implementation from the paper and the official repository (no license)
at the revision pinned in the model card; no official source was copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.empirical_quantiles import empirical_quantiles
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._inputs import check_window

NORM_EPS = 1e-5  # instance-normalisation variance floor
BETA_CLAMP = 1e-6  # learned beta is clamped to [1e-6, 1 - 1e-6]
RATIO_EPS = 1e-4  # Eq. 22 denominator floor
RATIO_CLIP = 10.0  # Eq. 22 clipping bounds r_min = -10, r_max = 10
SPECTRUM_EPS = 1e-8  # Eqs. 7-8 and 69
DDIM_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Frequency Guided Denoiser
# ---------------------------------------------------------------------------


class ConditionalGuidedModule(nn.Module):
    """Deterministic anchor ``x0_freq = x_psi(c0)`` (Eqs. 15-20): ``[B, L, N] -> [B, H, N]``."""

    def __init__(self, seq_len: int, pred_len: int, bands: int, init_scale: float = 0.02) -> None:
        super().__init__()
        frequencies = seq_len // 2 + 1
        self.seq_len, self.bands = seq_len, bands
        self.gate_slope = nn.Parameter(torch.zeros(frequencies))  # a in Eq. 17
        self.gate_bias = nn.Parameter(torch.zeros(frequencies))  # b in Eq. 17
        sizes = [len(part) for part in torch.tensor_split(torch.arange(frequencies), bands)]
        self.band_gains = nn.ParameterList(
            nn.Parameter(torch.view_as_real(torch.randn(size, dtype=torch.cfloat) * init_scale)) for size in sizes
        )
        self.projection = nn.Linear(seq_len, pred_len)

    def frequency_gate(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Eqs. 16-17: ``sigmoid(a * log(1 + mean_ch |C_f|) + b)`` -> ``[B, F]``."""
        energy = torch.log1p(spectrum.abs().mean(dim=1))
        return torch.sigmoid(self.gate_slope * energy + self.gate_bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(history.transpose(1, 2), dim=-1)  # [B, N, F]
        spectrum = spectrum * self.frequency_gate(spectrum)[:, None, :]
        parts = torch.tensor_split(spectrum, self.bands, dim=-1)
        spectrum = torch.cat(
            [part * torch.view_as_complex(gain) for part, gain in zip(parts, self.band_gains)], dim=-1
        )  # Eqs. 18-19
        filtered = torch.fft.irfft(spectrum, n=self.seq_len, dim=-1)  # Eq. 20
        return self.projection(filtered).transpose(1, 2)


class FiLM(nn.Module):
    """``(1 + gamma_t) h + beta_t`` with ``[gamma_t, beta_t]`` from a step embedding table."""

    def __init__(self, steps: int, hidden: int) -> None:
        super().__init__()
        self.table = nn.Embedding(steps, 2 * hidden)
        nn.init.uniform_(self.table.weight, -0.02, 0.02)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.table(t)[:, None, :].chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class SpectralConditionedDenoiser(nn.Module):
    """``x0_diff = x_theta(x_t, t, c0)`` guided by the history's spectral distortion (Eqs. 21-25)."""

    def __init__(self, steps: int, channels: int, pred_len: int, hidden: int) -> None:
        super().__init__()
        bottleneck = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.Linear(channels, bottleneck), nn.SiLU(), nn.Linear(bottleneck, channels), nn.Sigmoid()
        )
        self.raw_input = nn.Linear(pred_len, hidden)
        self.guided_input = nn.Linear(pred_len, hidden)
        self.film1 = FiLM(steps + 1, hidden)
        self.film2 = FiLM(steps + 1, hidden)
        self.refine = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, pred_len)

    @staticmethod
    def spectral_distortion(clean: torch.Tensor, corrupted: torch.Tensor) -> torch.Tensor:
        """Eq. 22 as in the official code: clipped relative magnitude change, tanh, mean over bins."""
        clean_mag = torch.fft.rfft(clean, dim=1).abs()
        noisy_mag = torch.fft.rfft(corrupted, dim=1).abs()
        ratio = ((noisy_mag - clean_mag) / (clean_mag + RATIO_EPS)).clamp(-RATIO_CLIP, RATIO_CLIP)
        return torch.tanh(ratio).mean(dim=1)  # [B, N]

    def forward(self, history, noisy_history, x_t, t):
        gate = self.gate(self.spectral_distortion(history, noisy_history))[:, :, None]  # Eq. 23, [B, N, 1]
        y = x_t.transpose(1, 2)  # each variable trajectory as an H-vector
        h = self.raw_input(y) + self.guided_input(y * gate)  # Eq. 24 and h0
        h = F.silu(self.film1(h, t))  # Eq. 25
        h = F.silu(self.film2(self.refine(h), t))
        return self.head(h).transpose(1, 2)


class FrequencyGuidedDenoiser(nn.Module):
    """Eq. 26: ``x0_hat = w x0_diff + (1 - w) x0_freq`` with a learned ``w = sigmoid(raw)``."""

    def __init__(self, seq_len, pred_len, channels, steps, hidden, bands) -> None:
        super().__init__()
        self.anchor = ConditionalGuidedModule(seq_len, pred_len, bands)
        self.denoiser = SpectralConditionedDenoiser(steps, channels, pred_len, hidden)
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, history, noisy_history, x_t, t):
        weight = torch.sigmoid(self.fusion_logit)
        return weight * self.denoiser(history, noisy_history, x_t, t) + (1.0 - weight) * self.anchor(history)


# ---------------------------------------------------------------------------
# Spectral Trajectory Scheduler objectives
# ---------------------------------------------------------------------------


def spectral_flatness(x: torch.Tensor) -> torch.Tensor:
    """Eq. 69 on the power spectrum pooled over batch and variables (official code): scalar."""
    power = torch.fft.rfft(x, dim=1).abs().square().mean(dim=(0, 2)) + SPECTRUM_EPS
    return torch.exp(power.log().mean() - power.mean().log())


def endpoint_kl(x_terminal: torch.Tensor) -> torch.Tensor:
    """Eqs. 7-8: KL of the variable-averaged spectral mass (DC bin excluded, as coded) to uniform."""
    power = (torch.fft.rfft(x_terminal, dim=1).abs().square().mean(dim=2) + SPECTRUM_EPS)[:, 1:]
    mass = power / (power.sum(dim=1, keepdim=True) + SPECTRUM_EPS)
    bins = mass.shape[1]
    return (mass * (mass.clamp_min(SPECTRUM_EPS).log() + math.log(bins))).sum(dim=1).mean()


class Model(nn.Module):
    """StaTS with a learned schedule, an x0-predicting FGD and DDIM sampling.

    ``forward`` returns empirical quantiles ``[B, pred_len, enc_in, K]`` of
    ``num_samples`` trajectories.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        quantile_levels: list[float],
        hidden: int = 256,
        bands: int = 2,
        diffusion_steps: int = 50,
        beta_start: float = 1e-5,
        beta_end: float = 0.1,
        schedule_epochs: int = 3,
        stage_lr: float = 1e-3,
        stage_weight_decay: float = 5e-4,
        stage_grad_clip: float = 5.0,
        lambda_smooth: float = 5.0,
        lambda_init: float = 0.5,
        lambda_end: float = 0.5,
        lambda_bar: float = 5e-3,
        lambda_prog: float = 0.5,
        lambda_obj: float = 0.01,
        num_samples: int = 100,
        sample_batch_size: int = 100,
    ) -> None:
        super().__init__()
        levels = list(quantile_levels)
        if not levels or any(not 0.0 < q < 1.0 for q in levels) or levels != sorted(set(levels)):
            raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
        self.output_type = "quantile"
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.num_steps = diffusion_steps
        self.schedule_epochs = schedule_epochs
        self.stage_lr, self.stage_weight_decay, self.stage_grad_clip = stage_lr, stage_weight_decay, stage_grad_clip
        self.weights = {
            "smooth": lambda_smooth,
            "init": lambda_init,
            "end": lambda_end,
            "bar": lambda_bar,
            "prog": lambda_prog,
            "obj": lambda_obj,
        }
        self.num_samples, self.sample_batch_size = num_samples, sample_batch_size
        self.denoiser = FrequencyGuidedDenoiser(seq_len, pred_len, enc_in, diffusion_steps, hidden, bands)
        # STS: beta(t) for the 0-based steps t = 0 .. T-1, linear initialisation.
        self.beta = nn.Parameter(torch.linspace(beta_start, beta_end, diffusion_steps))
        self.register_buffer("quantile_levels", torch.tensor(levels), persistent=False)
        self.register_buffer("schedule_frozen", torch.tensor(False))

    # -- schedule -------------------------------------------------------------
    def betas(self) -> torch.Tensor:
        return self.beta.clamp(BETA_CLAMP, 1.0 - BETA_CLAMP)

    def alpha_bar(self) -> torch.Tensor:
        """``abar_t = prod_{s <= t} (1 - beta_s)`` (Proposition 1)."""
        return torch.cumprod(1.0 - self.betas(), dim=0)

    # -- objectives -----------------------------------------------------------
    def _corrupt(self, history_n, target_n, t, abar):
        a = abar[t].view(-1, 1, 1)
        noise = torch.randn_like(target_n)
        noisy_target = a.sqrt() * target_n + (1 - a).sqrt() * noise  # Eq. 1 closed form
        noisy_history = a.sqrt() * history_n + (1 - a).sqrt() * torch.randn_like(history_n)
        return noisy_target, noisy_history, noise

    def objective_terms(self, history: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        """L_obj (Eq. 3) and the STS regularisers (Eqs. 6-12) for one batch."""
        history_n, mean, std = standardize(history, NORM_EPS)
        target_n = (future - mean) / std
        batch = history.shape[0]
        abar = self.alpha_bar()
        t = torch.randint(0, self.num_steps, (batch,), device=history.device)
        noisy_target, noisy_history, noise = self._corrupt(history_n, target_n, t, abar)
        prediction = destandardize(self.denoiser(history_n, noisy_history, noisy_target, t), mean, std)
        terms = {"obj": F.mse_loss(prediction, future)}  # official: de-normalised x0 regression
        betas = self.betas()
        terminal = abar[-1]
        x_terminal = terminal.sqrt() * target_n + (1 - terminal).sqrt() * torch.randn_like(target_n)
        terms["end"] = endpoint_kl(x_terminal)  # Eqs. 7-8
        terms["bar"] = -torch.log(betas[1:]).mean()  # Eq. 6
        terms["init"] = betas[0].square()  # Eq. 12, L_init
        terms["smooth"] = (betas[1:] - betas[:-1]).square().sum()  # Eq. 11
        end = terminal.sqrt() * target_n + (1 - terminal).sqrt() * noise
        gamma = (t.float() / (self.num_steps - 1)).view(-1, 1, 1)
        target_flatness = (1 - gamma) * spectral_flatness(target_n) + gamma * spectral_flatness(end)  # Eq. 10
        terms["prog"] = (spectral_flatness(noisy_target) - target_flatness).square().mean()  # Eq. 11
        return terms

    def schedule_loss(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Eq. 13: weighted STS objective (App. C.6 weights)."""
        terms = self.objective_terms(history, future)
        return sum(self.weights[name] * value for name, value in terms.items())

    def denoiser_loss(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """L_obj (Eq. 3) for FGD under the current schedule."""
        with torch.no_grad():
            abar = self.alpha_bar()
        history_n, mean, std = standardize(history, NORM_EPS)
        target_n = (future - mean) / std
        t = torch.randint(0, self.num_steps, (history.shape[0],), device=history.device)
        noisy_target, noisy_history, _ = self._corrupt(history_n, target_n, t, abar)
        prediction = destandardize(self.denoiser(history_n, noisy_history, noisy_target, t), mean, std)
        return F.mse_loss(prediction, future)

    # -- two-stage training (Sec. 3.2) ----------------------------------------
    def freeze_schedule(self) -> None:
        self.beta.requires_grad_(False)
        self.schedule_frozen.fill_(True)

    def pretrain(self, train_loader, device) -> None:
        """Stage I: ``schedule_epochs`` x (one FGD epoch with beta fixed, one STS epoch with FGD fixed)."""
        if bool(self.schedule_frozen):
            self.freeze_schedule()
            return
        self.to(device)
        self.train()
        denoiser_optim = torch.optim.Adam(
            self.denoiser.parameters(), lr=self.stage_lr, weight_decay=self.stage_weight_decay
        )
        schedule_optim = torch.optim.Adam([self.beta], lr=self.stage_lr, weight_decay=self.stage_weight_decay)
        for _ in range(self.schedule_epochs):
            self.beta.requires_grad_(False)
            self.denoiser.requires_grad_(True)
            for batch in train_loader:
                x, y = batch[0].float().to(device), batch[1].float().to(device)
                denoiser_optim.zero_grad()
                loss = self.denoiser_loss(x, y[:, -self.pred_len :])
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.stage_grad_clip)
                denoiser_optim.step()
            self.beta.requires_grad_(True)
            self.denoiser.requires_grad_(False)
            for batch in train_loader:
                x, y = batch[0].float().to(device), batch[1].float().to(device)
                schedule_optim.zero_grad()
                self.schedule_loss(x, y[:, -self.pred_len :]).backward()
                schedule_optim.step()
        self.denoiser.requires_grad_(True)
        self.freeze_schedule()

    # -- sampling -------------------------------------------------------------
    def ddim_step(self, x_t, x0_hat, step: int, abar: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM (eta = 0) update from an x0 prediction (official ``impute``)."""
        if step == 0:
            return x0_hat
        noise = (x_t / abar[step].sqrt() - x0_hat) / torch.sqrt(1.0 / abar[step] - 1.0)
        previous = abar[step - 1]
        return previous.sqrt() * x0_hat + torch.sqrt((1.0 - previous).clamp_min(DDIM_FLOOR)) * noise

    @torch.no_grad()
    def sample(self, x_enc: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """Trajectories on the input scale: ``[S, B, pred_len, N]``."""
        total = num_samples or self.num_samples
        history_n, mean, std = standardize(x_enc, NORM_EPS)
        abar = self.alpha_bar()
        chunks, done = [], 0
        while done < total:
            count = min(self.sample_batch_size, total - done)
            history = history_n.repeat(count, 1, 1)
            x = torch.randn(history.shape[0], self.pred_len, self.enc_in, device=x_enc.device, dtype=x_enc.dtype)
            for step in reversed(range(self.num_steps)):
                a = abar[step]
                noisy_history = a.sqrt() * history + (1 - a).sqrt() * torch.randn_like(history)  # fresh c_t
                t = torch.full((history.shape[0],), step, device=x_enc.device, dtype=torch.long)
                x = self.ddim_step(x, self.denoiser(history, noisy_history, x, t), step, abar)
            chunks.append(destandardize(x.view(count, *x_enc.shape[:1], self.pred_len, self.enc_in), mean, std))
            done += count
        return torch.cat(chunks, dim=0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        samples = self.sample(x_enc)
        return empirical_quantiles(samples, self.quantile_levels.to(samples.dtype))

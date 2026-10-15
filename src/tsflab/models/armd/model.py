"""ARMD: auto-regressive moving diffusion for time series forecasting.

Paper: Gao, Cao and Chen, "Auto-Regressive Moving Diffusion Models for Time Series
Forecasting" (arXiv 2412.09328).

The future series ``X^0 = x[1:T]`` is the initial state of the diffusion and the
history ``x[-T+1:0]`` its final state. The intermediate state at step ``t`` is the
length-T window slid ``t`` steps from the future toward the history (Eq. 1-2), so
the diffusion "noise" is the deterministic evolution trend

    z_t = (X^t / sqrt(abar_t) - X^0) / sqrt(1 / abar_t - 1)                (Eq. 3)

A linear devolution network predicts ``X^0`` from ``X^t`` (Eq. 4-5)

    X0_hat = (W(t) X^t + (1 - b W(t)) D) / (1 + c W(t))^d,  D = Linear(X^t)

with the reference constants b = 2, c = -1, d = 1/2 and a learnable schedule W
initialised to a linear-schedule cumulative product. Training minimises the
weighted L1 distance between ``z_t`` and ``z_hat_t`` (Eq. 6-7); forecasting starts
from the history window and applies deterministic DDIM updates (Eq. 9-10) with
``z_hat`` in place of the predicted noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


def linear_beta_schedule(steps: int) -> torch.Tensor:
    scale = 1000 / steps
    return torch.linspace(scale * 1e-4, scale * 0.02, steps, dtype=torch.float64)


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos((grid / steps + offset) / (1 + offset) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    return torch.clip(1 - cumulative[1:] / cumulative[:-1], 0, 0.999)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        sampling_steps: int = 1,
        loss_type: str = "l1",
        beta_schedule: str = "cosine",
        w_grad: bool = True,
    ) -> None:
        super().__init__()
        if seq_len < pred_len:
            raise ValueError("ARMD slides a window of pred_len steps, so seq_len >= pred_len")
        if sampling_steps > pred_len:
            raise ValueError("sampling_steps must not exceed the diffusion length pred_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.sampling_steps, self.loss_type = sampling_steps, loss_type
        steps = pred_len  # the maximum diffusion step equals the series length

        betas = (linear_beta_schedule if beta_schedule == "linear" else cosine_beta_schedule)(steps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)

        def floats(value):
            return value.to(torch.float32)

        self.register_buffer("alphas_cumprod", floats(cumulative))
        self.register_buffer("sqrt_alphas_cumprod", floats(cumulative.sqrt()))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", floats((1 - cumulative).sqrt()))
        self.register_buffer("sqrt_recip_alphas_cumprod", floats(cumulative.rsqrt()))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", floats((1 / cumulative - 1).sqrt()))
        # loss re-weighting of the reference training loop (per diffusion step)
        self.register_buffer("loss_weight", floats(alphas.sqrt() * (1 - cumulative).sqrt() / betas / 100))

        # devolution network: one linear map over the time axis shared by channels,
        # a learnable step weight W(t), and the fixed input-deviation scale per step
        self.temporal = nn.Linear(pred_len, pred_len)
        step_weight = torch.cumprod(1.0 - linear_beta_schedule(steps), dim=0)
        self.step_weight = nn.Parameter(floats(step_weight), requires_grad=w_grad)
        self.register_buffer("deviation_scale", floats(1.0 - cosine_beta_schedule(steps)))

    # -------------------------------------------------------------- devolution
    def devolve(self, state: torch.Tensor, step: int, perturb: bool = False) -> torch.Tensor:
        """Predict ``X^0`` from the state ``X^t`` (Eq. 4-5); ``step`` is 0-based."""
        if perturb:  # small training-time deviation of the network input
            state = state + self.deviation_scale[step] * torch.randn_like(state)
        distance = self.temporal(state.transpose(1, 2)).transpose(1, 2)
        weight = self.step_weight[step]
        return (weight * state + (1 - 2 * weight) * distance) / (1 - weight).sqrt()

    def _trend(self, state: torch.Tensor, start: torch.Tensor, step: int) -> torch.Tensor:
        """Evolution trend ``z`` between a state and a (predicted) future (Eq. 3, 6)."""
        return (state - self.sqrt_alphas_cumprod[step] * start) / self.sqrt_one_minus_alphas_cumprod[step]

    # ------------------------------------------------------------------ training
    def intermediate_state(self, series: torch.Tensor, step: int) -> torch.Tensor:
        """Window of ``series`` (history then future, length 2T) slid ``step + 1`` back (Eq. 2).

        Step ``T - 1`` returns the history window and step 0 the window one step
        before the future.
        """
        length = self.pred_len
        shift = step + 1
        return series[:, length - shift : 2 * length - shift]

    def training_loss(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Weighted distance between true and predicted evolution trends (Eq. 7).

        ``history`` is ``(B, seq_len, C)`` and ``future`` ``(B, pred_len, C)``; one
        diffusion step is drawn per call and shared by the batch.
        """
        length = self.pred_len
        if history.shape[1] < length or future.shape[1] != length:
            raise ValueError("history needs at least pred_len steps and future exactly pred_len")
        series = torch.cat([history[:, -length:], future], dim=1)  # (B, 2T, C)
        step = int(torch.randint(0, length, (1,)))
        state = self.intermediate_state(series, step)
        predicted = self.devolve(state, step, perturb=True)
        true_trend = self._trend(state, future, step)
        predicted_trend = self._trend(state, predicted, step)
        distance = (F.l1_loss if self.loss_type == "l1" else F.mse_loss)(predicted_trend, true_trend, reduction="none")
        return (distance.flatten(1).mean(dim=1) * self.loss_weight[step]).mean()

    # ------------------------------------------------------------------ forecast
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        total = self.pred_len
        state = x_enc[:, -total:, :]
        times = torch.linspace(-1, total - 1, steps=self.sampling_steps + 1).int().tolist()
        times.reverse()
        for time, time_next in zip(times[:-1], times[1:]):
            start = self.devolve(state, time)
            if time_next < 0:
                return start
            trend = (self.sqrt_recip_alphas_cumprod[time] * state - start) / self.sqrt_recipm1_alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            state = start * alpha_next.sqrt() + (1 - alpha_next).sqrt() * trend
        return state

"""DiPE-Linear: static frequency and time attention, then a long FFT convolution.

Per channel (Section 3 of the paper), with ``x`` the instance-normalized history:

    z_sfa = irfft(theta_sfa * rfft(x))                         (1)
    z_sta = theta_sta * z_sfa                                  (2)
    y     = irfft(theta_ifm * rfft(pad(z_sta)) + beta_ifm)[-L':]   (3)-(5)

``theta_*`` are low-rank shared: ``num_experts`` weight sets are mixed per
channel by ``softmax(R / tau)`` (Eqs. 7-8). The SFA-weighted frequency loss
(Eqs. 9-11) is exposed through ``sfa_loss`` and the runner's training objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.fft_extrapolation_conv import FFTExtrapolationConv
from tsflab.models._components.weight_set_router import WeightSetRouter, mix_weight_sets


def _check_arguments(
    seq_len: int, pred_len: int, enc_in: int, num_experts: int, loss_alpha: float, dropout: float
) -> None:
    if min(pred_len, enc_in, num_experts) < 1 or seq_len < 2:
        raise ValueError("seq_len must be >= 2 and other sizes positive")
    if not 0 <= loss_alpha <= 1:
        raise ValueError("loss_alpha must lie in [0, 1]")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must lie in [0, 1)")


def _check_schedule(temperature_start: float, temperature_end: float, anneal_epochs: int) -> None:
    if not 0 < temperature_end <= temperature_start or anneal_epochs < 1:
        raise ValueError("need 0 < temperature_end <= temperature_start and anneal_epochs >= 1")


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_experts: int = 1,
        use_revin: bool = True,
        use_time_w: bool = True,
        use_freq_w: bool = True,
        loss_alpha: float = 0.9,
        dropout: float = 0.1,
        temperature_start: float = 30.0,
        temperature_end: float = 1.0,
        anneal_epochs: int = 10,
    ) -> None:
        super().__init__()
        _check_arguments(seq_len, pred_len, enc_in, num_experts, loss_alpha, dropout)
        _check_schedule(temperature_start, temperature_end, anneal_epochs)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.num_experts = num_experts
        self.use_revin, self.use_time_w, self.use_freq_w = use_revin, use_time_w, use_freq_w
        self.loss_alpha = float(loss_alpha)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.anneal_epochs = anneal_epochs

        self.router = WeightSetRouter(num_experts, enc_in) if num_experts > 1 else None
        # theta_sfa (one weight per rfft bin) and theta_sta (one per time step).
        self.freq_weight = nn.Parameter(torch.ones(num_experts, seq_len // 2 + 1)) if use_freq_w else None
        self.time_weight = nn.Parameter(torch.ones(num_experts, seq_len)) if use_time_w else None
        self.dropout = nn.Dropout(dropout)
        self.mapping = FFTExtrapolationConv(seq_len, pred_len, num_experts)
        # Training-progress state for temperature annealing (see ``training_setup``).
        self.register_buffer("steps_per_epoch", torch.zeros((), dtype=torch.long))
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ schedule
    def temperature(self) -> float:
        """Softmax temperature: linear anneal while training, ``temperature_end`` in eval."""
        steps = int(self.steps_per_epoch)
        if not self.training or steps == 0:
            return self.temperature_end
        epoch = int(self.train_steps) // steps
        if epoch >= self.anneal_epochs:
            return self.temperature_end
        span = self.temperature_start - self.temperature_end
        return self.temperature_start - span * epoch / self.anneal_epochs

    def _routing(self) -> torch.Tensor | None:
        return None if self.router is None else self.router(self.temperature())

    @staticmethod
    def _mix(weights: torch.Tensor, routing: torch.Tensor | None) -> torch.Tensor:
        # (F,) shared by all channels, or (C, F) after low-rank mixing.
        return weights[0] if routing is None else mix_weight_sets(weights, routing)

    # ------------------------------------------------------------------- forward
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        x = x_enc.transpose(1, 2)  # [B, C, L]
        if self.use_revin:
            mean = x.mean(dim=-1, keepdim=True).detach()
            std = x.std(dim=-1, keepdim=True).detach().clamp(min=1e-7)
            x = (x - mean) / std
        routing = self._routing()
        if self.freq_weight is not None:  # static frequential attention, Eq. (1)
            weight = self._mix(self.freq_weight, routing)
            x = torch.fft.irfft(torch.fft.rfft(x) * weight, n=self.seq_len)
        x = self.dropout(x)
        if self.time_weight is not None:  # static temporal attention, Eq. (2)
            x = x * self._mix(self.time_weight, routing)
        y = self.mapping(x, routing)  # independent frequential mapping, Eqs. (3)-(5)
        if self.use_revin:
            y = y * std + mean
        return y.transpose(1, 2)

    # ---------------------------------------------------------------------- loss
    def sfa_loss(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SFALoss, Eq. (11): ``alpha * L_F + (1 - alpha) * L_T`` on ``[B, H, C]``."""
        if forecast.shape != target.shape or forecast.shape[1] != self.pred_len:
            raise ValueError("forecast and target must both be [batch, pred_len, channels]")
        channels = forecast.shape[-1]
        time_loss = (forecast - target).square().mean()  # Eq. (10)
        if self.loss_alpha == 0:
            return time_loss
        y, y_hat = target.transpose(1, 2), forecast.transpose(1, 2)
        diff = (torch.fft.rfft(y, norm="ortho") - torch.fft.rfft(y_hat, norm="ortho")).abs()
        weight = 1.0 if self.freq_weight is None else self._loss_weight(diff.shape[-1], channels)
        freq_loss = (weight * diff).mean()  # Eq. (9)
        return self.loss_alpha * freq_loss + (1 - self.loss_alpha) * time_loss

    def _loss_weight(self, bins: int, channels: int) -> torch.Tensor:
        """theta_sfa normalized and aligned to the ``bins`` of the horizon spectrum."""
        with torch.no_grad():  # theta_sfa is detached in the loss
            weight = self._mix(self.freq_weight, self._routing()).detach()
            weight = weight / weight.abs().mean(dim=-1, keepdim=True).clamp_min(1e-12)
            weight = weight[..., :bins]  # horizon spectrum is shorter or longer than the history's
            if weight.shape[-1] < bins:
                weight = nn.functional.pad(weight, (0, bins - weight.shape[-1]))
            if weight.ndim == 2:  # channel-specific weights; keep the trailing channels
                weight = weight[-channels:]
        return weight

    def training_objective(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss for one training pass; advances the annealing step counter."""
        loss = self.sfa_loss(forecast, target)
        self.train_steps += 1
        return loss

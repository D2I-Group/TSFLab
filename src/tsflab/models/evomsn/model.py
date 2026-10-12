"""EvoMSN (offline): multi-scale slice normalization around a shared DLinear backbone.

Independent implementation from Sections 3.1-3.3 (Eqs. 1-10) of Qin et al.,
"Evolving Multi-Scale Normalization for Time Series Forecasting under Distribution
Shifts" (arXiv 2409.19718), after reading the pinned official code
(``qindalin/EvoMSN`` at ``2278adeb``) to resolve omissions; nothing is copied. Only
the offline two-stage training is implemented; the online alternating updates
need a streaming protocol the catalog runner does not provide.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._inputs import check_window

_EPS = 1e-5


def admissible_frequencies(seq_len: int, pred_len: int) -> range:
    """rFFT bins eligible as global periods: DC and periods beyond ~2 * pred_len are excluded."""
    return range(seq_len // (2 * pred_len) + 1, seq_len // 2 + 1)


def default_periods(seq_len: int, pred_len: int, top_k: int) -> tuple[int, ...]:
    """Data-free placeholder used until :meth:`Model.pretrain` selects periods: the
    ``top_k`` lowest admissible frequencies (longest periods)."""
    frequencies = list(admissible_frequencies(seq_len, pred_len))[:top_k]
    if len(frequencies) < top_k:
        raise ValueError(f"seq_len={seq_len} admits fewer than top_k={top_k} periods")
    return tuple(seq_len // f for f in frequencies)


def slice_pad(series: torch.Tensor, period: int) -> torch.Tensor:
    """Eq. (2) padding + slicing: append the trailing segment of the series so the length is
    a multiple of ``period``, then reshape ``[B, T, C] -> [B, ceil(T / p), p, C]``."""
    length = series.shape[1]
    missing = -length % period
    if missing:
        series = torch.cat([series, series[:, length - missing:]], dim=1)
    return series.reshape(series.shape[0], -1, period, series.shape[2])


def slice_statistics(series: torch.Tensor, period: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (3)-(4): per-slice mean and (population) standard deviation, ``[B, S, C]`` each."""
    slices = slice_pad(series, period)
    mean = slices.mean(dim=2)
    std = slices.var(dim=2, unbiased=False).sqrt()
    return mean, std


class StatisticsBranch(nn.Module):
    """Two-layer perceptron over slice statistics and the raw window (Eq. 5)."""

    def __init__(self, seq_len: int, in_slices: int, out_slices: int, hidden: int, mode: str) -> None:
        super().__init__()
        self.statistics = nn.Linear(in_slices, hidden)
        self.raw = nn.Linear(seq_len, hidden)
        self.output = nn.Linear(2 * hidden, out_slices)
        self.hidden_activation = nn.Tanh() if mode == "mean" else nn.ReLU()
        self.final_activation = nn.Identity() if mode == "mean" else nn.ReLU()

    def forward(self, stats: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # [B, S, C], [B, L, C] -> [B, S_out, C]; the maps act along time per channel.
        hidden = torch.cat([self.statistics(stats.transpose(1, 2)), self.raw(raw.transpose(1, 2))], dim=-1)
        out = self.output(self.hidden_activation(hidden))
        return self.final_activation(out).transpose(1, 2)


class ScaleStatisticsPredictor(nn.Module):
    """``f_omega_i`` / ``f_theta_i`` for one period: future slice means and standard deviations.

    The mean branch works on statistics centered by the window mean and adds it back
    with learnable per-channel weights (official clarification); the standard
    deviation branch ends in ReLU.
    """

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, period: int, hidden: int) -> None:
        super().__init__()
        self.period = period
        in_slices = math.ceil(seq_len / period)
        out_slices = math.ceil(pred_len / period)
        self.mean_branch = StatisticsBranch(seq_len, in_slices, out_slices, hidden, "mean")
        self.std_branch = StatisticsBranch(seq_len, in_slices, out_slices, hidden, "std")
        self.mix = nn.Parameter(torch.ones(2, enc_in))

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window_mean = x.mean(dim=1, keepdim=True)
        future_mean = self.mean_branch(mean - window_mean, x - window_mean) * self.mix[0] + window_mean * self.mix[1]
        return future_mean, self.std_branch(std, x)


class Model(nn.Module):
    """Multi-scale slice normalization, shared DLinear backbone, amplitude-weighted ensemble."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        top_k: int = 4,
        periods: Sequence[int] | None = None,
        stat_hidden: int = 512,
        stat_pretrain_epochs: int = 5,
        stat_lr: float = 1e-4,
        kernel_size: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.stat_hidden = stat_hidden
        self.stat_pretrain_epochs = stat_pretrain_epochs
        self.stat_lr = stat_lr
        # Periods given explicitly are kept; otherwise pretrain() selects them from data (Eq. 1).
        self.select_periods = periods is None
        if periods is None:
            periods = default_periods(seq_len, pred_len, top_k)
        periods = tuple(int(p) for p in periods)
        if len(periods) != top_k or min(periods) < 1 or max(periods) > seq_len:
            raise ValueError("periods must hold top_k integers in [1, seq_len]")
        self.backbone = DLinearBackbone(enc_in, seq_len, pred_len, kernel_size, individual)
        self._build_statistics(periods)
        self._statistics_frozen = False

    def _build_statistics(self, periods: tuple[int, ...]) -> None:
        self.periods = periods
        self.statistics = nn.ModuleList(
            ScaleStatisticsPredictor(self.seq_len, self.pred_len, self.enc_in, p, self.stat_hidden)
            for p in periods
        )

    def _validate(self, x: torch.Tensor) -> None:
        check_window(x, self.seq_len, self.enc_in)

    # ------------------------------------------------------------------ Eq. 1
    @torch.no_grad()
    def global_periods(self, train_loader, device) -> tuple[int, ...]:
        """Eq. (1): top-k frequencies of the amplitude spectrum averaged over training
        windows and channels, excluding DC and frequencies below ``seq_len // (2 H) + 1``."""
        total, batches = None, 0
        for batch in train_loader:
            x = batch[0].float().to(device)
            amps = torch.fft.rfft(x, dim=1).abs().mean(dim=0).mean(dim=1)
            total = amps if total is None else total + amps
            batches += 1
        if total is None:
            raise ValueError("cannot select periods from an empty training loader")
        amps = total / batches
        amps[: admissible_frequencies(self.seq_len, self.pred_len).start] = 0
        top = amps.topk(len(self.periods)).indices
        return tuple(int(self.seq_len // f) for f in top.tolist())

    # --------------------------------------------------------------- Eqs. 2-10
    def predict_statistics(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Predicted future slice means and standard deviations for every period."""
        out = []
        for period, predictor in zip(self.periods, self.statistics):
            mean, std = slice_statistics(x, period)
            out.append(predictor(x, mean, std))
        return out

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (6): one slice-normalized copy of the window per period, ``[k, B, L, C]``."""
        copies = []
        for period in self.periods:
            slices = slice_pad(x, period)
            mean = slices.mean(dim=2, keepdim=True)
            std = slices.var(dim=2, unbiased=False, keepdim=True).sqrt()
            normalized = (slices - mean) / (std + _EPS)
            copies.append(normalized.reshape(x.shape[0], -1, self.enc_in)[:, : self.seq_len])
        return torch.stack(copies)

    def denormalize(self, outputs: torch.Tensor, statistics) -> torch.Tensor:
        """Eq. (8): ``y = y~ * (xi^ + eps) + phi^`` slice by slice, ``[k, B, H, C]``."""
        restored = []
        for output, period, (mean, std) in zip(outputs, self.periods, statistics):
            slices = slice_pad(output, period)
            slices = slices * (std.unsqueeze(2) + _EPS) + mean.unsqueeze(2)
            restored.append(slices.reshape(output.shape[0], -1, self.enc_in)[:, : self.pred_len])
        return torch.stack(restored)

    def ensemble_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Eqs. (9)-(10): local amplitudes of the window at the global periods, normalized
        over the k periods, ``[k, B, 1, C]``."""
        amps = torch.fft.rfft(x, dim=1).abs()
        index = torch.tensor([self.seq_len // p for p in self.periods], device=x.device)
        local = amps.index_select(1, index)  # [B, k, C]
        weights = local / local.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return weights.permute(1, 0, 2).unsqueeze(2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        self._validate(x_enc)
        batch = x_enc.shape[0]
        normalized = self.normalize(x_enc)  # [k, B, L, C]
        k = normalized.shape[0]
        outputs = self.backbone(normalized.reshape(k * batch, self.seq_len, self.enc_in))
        outputs = outputs.reshape(k, batch, self.pred_len, self.enc_in)  # Eq. (7), shared psi
        restored = self.denormalize(outputs, self.predict_statistics(x_enc))
        return (restored * self.ensemble_weights(x_enc)).sum(dim=0)

    # ---------------------------------------------------------------- Sec. 3.3
    def statistics_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Sum over periods of the MSE between predicted and true future slice statistics."""
        loss = x.new_zeros(())
        for period, (mean, std) in zip(self.periods, self.predict_statistics(x)):
            true_mean, true_std = slice_statistics(target, period)
            loss = loss + F.mse_loss(torch.cat([mean, std], -1), torch.cat([true_mean, true_std], -1))
        return loss

    def pretrain(self, train_loader, device) -> None:
        """Offline stage 1: select global periods (Eq. 1), train the statistics predictors on
        future slice statistics, then freeze them for the backbone stage."""
        if self._statistics_frozen:
            return
        if self.select_periods:
            periods = self.global_periods(train_loader, device)
            if periods != self.periods:
                self._build_statistics(periods)
        self.statistics.to(device).train()
        optimizer = torch.optim.Adam(self.statistics.parameters(), lr=self.stat_lr)
        for _ in range(self.stat_pretrain_epochs):
            for batch in train_loader:
                x = batch[0].float().to(device)
                y = batch[1].float().to(device)[:, -self.pred_len:, : self.enc_in]
                optimizer.zero_grad()
                self.statistics_loss(x, y).backward()
                optimizer.step()
        for parameter in self.statistics.parameters():
            parameter.requires_grad_(False)
        self._statistics_frozen = True

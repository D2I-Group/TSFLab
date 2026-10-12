"""SSCLForecaster: AutoCon autocorrelation-based contrastive learning with the
AutoConNet decomposition forecaster.

Independent implementation from Section 3 (Eqs. 1-8) and Appendix A (Eqs. 9-13,
Algorithm 1, channel independence) of Park et al., "Self-Supervised Contrastive
Learning for Long-term Forecasting" (ICLR 2024, arXiv 2402.02023), after reading
the pinned official code (``junwoopark92/Self-Supervised-Contrastive-Forecsating``
at ``f394705a``, MIT license) to resolve omissions; nothing is copied.

The AutoCon objective needs each training window's global position and the
autocorrelation of the whole training series. Both are recovered through the
catalog training contract: ``fit_autocorrelation`` (the ``training_setup``) rebuilds
the training series from one pass over the training loader, ordering windows by
the absolute start timestamp of their raw calendar marks, and ``window_index``
maps a batch's start timestamps back to those window positions.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding
from tsflab.models._components.marks import elapsed_minutes, encoder_timef_marks, tslib_time_feature_dimension
from tsflab.models._components.series_decomposition import EdgePaddedMovingAverage, SeriesDecomposition

# The official experiment passes temperature = base_temperature = 1.0 to AutoCon,
# so the (temperature / base_temperature) loss factor is 1.
TEMPERATURE = 1.0
WindowNorm = Literal["last", "mean", "revin", "decomp"]


# --------------------------------------------------------------------------- time


def mark_minutes(marks: torch.Tensor) -> torch.Tensor:
    """Absolute minute count from raw ``[..., 6]`` marks ``[year, month, day, weekday, hour, minute]``."""
    if marks.shape[-1] != 6:
        raise ValueError("AutoCon needs raw six-column calendar marks [year, month, day, weekday, hour, minute]")
    return elapsed_minutes(marks)


def autocorrelation(series: torch.Tensor) -> torch.Tensor:
    """Eq. (1) on a finite series: biased sample ACF of ``[T, C]`` for lags ``0..T-1``,
    returned as ``[C, T]`` (statsmodels ``acf`` with ``adjusted=False``)."""
    x = series.double() - series.double().mean(0, keepdim=True)
    length = x.shape[0]
    spectrum = torch.fft.rfft(x, n=2 * length, dim=0)
    autocov = torch.fft.irfft(spectrum * spectrum.conj(), n=2 * length, dim=0)[:length] / length
    variance = autocov[:1]
    acf = torch.where(variance > 0, autocov / variance.clamp_min(1e-300), torch.zeros_like(autocov))
    acf[0] = 1.0
    return acf.T.float()


# ------------------------------------------------------------------------ encoder


class DilatedConvBlock(nn.Module):
    """Residual block of the TCN encoder: GELU -> same-padded dilated conv, twice,
    plus an identity (or 1x1-projected) skip."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, final: bool, kernel_size: int = 3) -> None:
        super().__init__()
        receptive = (kernel_size - 1) * dilation + 1
        self.trim = 1 if receptive % 2 == 0 else 0
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=receptive // 2, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=receptive // 2, dilation=dilation)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def _same(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        out = conv(x)
        return out[..., : out.shape[-1] - self.trim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        return self._same(self.conv2, F.gelu(self._same(self.conv1, F.gelu(x)))) + residual


class DilatedConvEncoder(nn.Module):
    """Eq. (6) encoder: stacked dilated residual conv blocks (dilation ``2**i``) over ``[B, D, T]``."""

    def __init__(self, in_channels: int, channels: list[int], kernel_size: int = 3) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                DilatedConvBlock(
                    channels[i - 1] if i else in_channels, channels[i], 2**i, i == len(channels) - 1, kernel_size
                )
                for i in range(len(channels))
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# -------------------------------------------------------------------------- AutoCon


def contrastive_term(logits: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    """Autocorrelation-weighted contrastive loss per anchor, as in the official code.

    ``logits`` ``[G, N, N]`` are similarities over ``tau``; ``relation`` ``[G, N, N]``
    holds ``r = |R_SS(lag)|`` (Eq. 2). For anchor ``i`` the positives are the other
    members with the highest ``r`` (plus members sharing its global position,
    ``r = 1``); every other member forms the denominator; each positive is weighted
    by ``r``. Returns ``[G, N]``.
    """
    n = logits.shape[-1]
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    not_self = 1.0 - torch.eye(n, device=logits.device, dtype=logits.dtype)
    same = relation == 1.0
    masked = relation.masked_fill(same, float("-inf"))
    positive = (masked == masked.max(dim=2, keepdim=True).values).to(logits.dtype) + not_self * same.to(logits.dtype)
    log_prob = logits - torch.log((torch.exp(logits) * not_self).sum(2, keepdim=True))
    return -(relation * positive * log_prob).sum(2) / positive.sum(2)


def local_autocon(features: torch.Tensor, acf: torch.Tensor, channels: int) -> torch.Tensor:
    """Local term: ``T // 3`` random time steps of each window ``[BC, T, D]`` contrasted
    with lags equal to their in-window distance. Returns ``[BC, T // 3]``."""
    rows, length, width = features.shape
    index = torch.rand(rows, length, device=features.device).argsort(-1)[:, : length // 3]
    selected = torch.gather(features, 1, index.unsqueeze(-1).expand(-1, -1, width))
    logits = selected @ selected.transpose(1, 2) / TEMPERATURE
    lag = (index.unsqueeze(1) - index.unsqueeze(2)).abs()
    per_row = acf[torch.arange(rows, device=features.device) % channels]  # rows are (batch, channel)
    relation = torch.gather(per_row, 1, lag.reshape(rows, -1)).view_as(logits)
    return contrastive_term(logits, relation)


def global_autocon(features: torch.Tensor, positions: torch.Tensor, acf: torch.Tensor) -> torch.Tensor:
    """Global term (Eqs. 2-3): max-pooled window representations of each channel
    contrasted across the batch with ``r = |R_SS(|t_i - t_j|)|``. Returns ``[C, B]``."""
    batch = positions.shape[0]
    pooled = features.max(dim=1).values  # max pooling along time
    channels = pooled.shape[0] // batch
    windows = pooled.view(batch, channels, -1).transpose(0, 1)  # [C, B, D]
    logits = windows @ windows.transpose(1, 2) / TEMPERATURE
    lag = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    return contrastive_term(logits, acf[:, lag])


# ---------------------------------------------------------------------------- model


def _collect_windows(train_loader):
    """Window start times, first rows, and the latest window ``(time, x, y)`` of a loader."""
    starts, first_rows, last = [], [], None
    for batch in train_loader:
        x, y, x_mark = batch[0], batch[1], batch[2]
        if x_mark is None:
            raise ValueError("SSCLForecaster's AutoCon objective needs calendar marks")
        times = mark_minutes(torch.as_tensor(x_mark)[:, 0])
        x, y = torch.as_tensor(x).float(), torch.as_tensor(y).float()
        starts.append(times)
        first_rows.append(x[:, 0])
        latest = int(times.argmax())
        if last is None or times[latest] > last[0]:
            last = (times[latest], x[latest], y[latest])
    return starts, first_rows, last


class Model(nn.Module):
    """AutoConNet (Fig. 4) for ``[B, seq_len, enc_in]`` histories, channel independent."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 16,
        d_ff: int = 16,
        e_layers: int = 2,
        dropout: float = 0.1,
        scales: tuple[int, ...] | list[int] = (96,),
        window_norm: WindowNorm = "last",
        autocon_lambda: float = 1.0,
        use_marks: bool = True,
        freq: str = "h",
    ) -> None:
        super().__init__()
        if seq_len < 6 or seq_len % 2:
            raise ValueError("seq_len must be even (smoothing kernel seq_len + 1) and at least 6")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.window_norm = window_norm
        self.autocon_lambda = autocon_lambda
        self.use_marks = use_marks
        self.freq = freq
        marks_width = tslib_time_feature_dimension(freq)
        # Long-term branch: Enc(X_norm, T) (Eq. 6) and the multi-scale MA decoder (Eq. 7).
        self.embedding = DataEmbedding(1, d_model, "timeF", freq, dropout, time_feature_dim=marks_width)
        self.encoder = DilatedConvEncoder(d_model, [d_model] * e_layers + [d_ff])
        self.repr_dropout = nn.Dropout(0.1)
        self.length_mlp = nn.Linear(seq_len, pred_len)
        self.scale_averages = nn.ModuleList(EdgePaddedMovingAverage(scale + 1) for scale in scales)
        self.channel_mlps = nn.ModuleList(nn.Linear(d_ff, 1) for _ in scales)
        # Short-term branch (Eq. 5).
        self.short_linear = nn.Linear(seq_len, pred_len)
        self.input_decomposition = SeriesDecomposition(25)
        # AutoCon state fitted from the training split (training_setup).
        self.register_buffer("acf", torch.zeros(enc_in, 0))
        self.register_buffer("window_times", torch.zeros(0, dtype=torch.long))

    # -- forecasting -------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Window normalization (Eq. 4 and the official variants) -> (short, long, state)."""
        if self.window_norm == "decomp":
            short, long = self.input_decomposition(x)
            return short, long, None
        if self.window_norm == "revin":
            mean = x.mean(1, keepdim=True).detach()
            scale = x.std(1, keepdim=True).detach() + 1e-5
            normed = (x - mean) / scale
            return normed, normed, (mean, scale)
        reference = x[:, -1:].detach() if self.window_norm == "last" else x.mean(1, keepdim=True).detach()
        return x - reference, x - reference, (reference, None)

    def denormalize(self, prediction: torch.Tensor, state: object) -> torch.Tensor:
        if state is None:
            return prediction
        reference, scale = state  # type: ignore[misc]
        return prediction * scale + reference if scale is not None else prediction + reference

    def calendar(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        return encoder_timef_marks(x_mark_enc, seq_len=self.seq_len, freq=self.freq, enabled=self.use_marks)

    def represent(self, long_input: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """Channel-independent representation ``v`` ``[B * C, L, d_ff]`` (rows ordered batch-major)."""
        batch, length, channels = long_input.shape
        series = long_input.transpose(1, 2).reshape(batch * channels, length, 1)
        marks = self.calendar(x_mark_enc)
        if marks is not None:
            marks = marks.repeat_interleave(channels, dim=0)
        tokens = self.embedding(series, marks).transpose(1, 2)  # [BC, D, L]
        return self.repr_dropout(self.encoder(tokens)).transpose(1, 2)

    def long_term(self, representation: torch.Tensor, batch: int) -> torch.Tensor:
        """Multi-scale MA decoder (Eq. 7, summed over scales as in the official code) -> ``[B, H, C]``."""
        hidden = self.length_mlp(F.gelu(representation.transpose(1, 2))).transpose(1, 2)  # [BC, H, D]
        trend = 0.0
        for average, mlp in zip(self.scale_averages, self.channel_mlps):
            trend = trend + average(mlp(F.gelu(average(hidden))))
        return trend.reshape(batch, -1, self.pred_len).transpose(1, 2)  # type: ignore[union-attr]

    def forecast_with_representation(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in})")
        short, long, state = self.normalize(x_enc)
        representation = self.represent(long, x_mark_enc)
        long_term = self.long_term(representation, x_enc.shape[0])
        short_term = self.short_linear(short.transpose(1, 2)).transpose(1, 2)
        return self.denormalize(short_term + long_term, state), representation

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        return self.forecast_with_representation(x_enc, x_mark_enc)[0]

    # -- AutoCon -----------------------------------------------------------------

    @torch.no_grad()
    def fit_autocorrelation(self, train_loader) -> None:
        """Appendix A.1 preprocessing: rebuild the training series, smooth it with a
        moving average of length ``seq_len + 1``, and store per-channel ``|ACF|``
        together with the sorted window start timestamps."""
        starts, first_rows, last = _collect_windows(train_loader)
        if last is None:
            raise ValueError("the training loader is empty")
        times = torch.cat(starts)
        order = times.argsort()
        times = times[order]
        if times.numel() > 1 and not bool((times[1:] > times[:-1]).all()):
            raise ValueError("AutoCon needs distinct window start timestamps in the calendar marks")
        _, last_x, last_y = last
        series = torch.cat((torch.cat(first_rows)[order], last_x[1:], last_y[-self.pred_len :]))
        smoothed = EdgePaddedMovingAverage(self.seq_len + 1)(series.unsqueeze(0)).squeeze(0)
        device = self.length_mlp.weight.device
        self.acf = autocorrelation(smoothed).abs().to(device)
        self.window_times = times.to(device)

    def window_index(self, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """Global position ``t`` of each window (Section 3) from its first timestamp."""
        if self.window_times.numel() == 0:
            raise RuntimeError("SSCLForecaster needs fit_autocorrelation (training_setup) before AutoCon")
        times = mark_minutes(x_mark_enc[:, 0]).to(self.window_times.device)
        index = torch.searchsorted(self.window_times, times).clamp_max(self.window_times.numel() - 1)
        if not bool((self.window_times[index] == times).all()):
            raise ValueError("batch windows are not part of the fitted training split")
        return index

    def autocon_loss(self, representation: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """``L_AutoCon``: mean of the local and global terms over channels and windows."""
        if x_mark_enc is None:
            raise ValueError("SSCLForecaster's AutoCon objective needs calendar marks")
        if self.acf.shape[-1] < self.seq_len:
            raise RuntimeError("SSCLForecaster needs fit_autocorrelation (training_setup) before AutoCon")
        batch = x_mark_enc.shape[0]
        features = F.normalize(representation, dim=-1)
        local = local_autocon(features, self.acf, self.enc_in).view(batch, self.enc_in, -1).mean(2).mean(1)
        if batch > 1:
            global_term = global_autocon(features, self.window_index(x_mark_enc), self.acf).mean(0)
        else:  # a single window has no partner
            global_term = torch.zeros_like(local)
        return ((local + global_term) / 2.0).mean()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The fitted AutoCon state has a data-dependent size; resize before loading.
        for name in ("acf", "window_times"):
            key = prefix + name
            if key in state_dict and state_dict[key].shape != self._buffers[name].shape:
                self._buffers[name] = torch.empty(
                    state_dict[key].shape, dtype=self._buffers[name].dtype, device=self._buffers[name].device
                )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


__all__ = ["Model", "autocorrelation", "contrastive_term", "global_autocon", "local_autocon", "mark_minutes"]

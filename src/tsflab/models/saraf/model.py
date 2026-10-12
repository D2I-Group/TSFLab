"""SARAF: stationarity-aware retrieval-augmented forecasting (local implementation).

Independent rewrite from Section 3 of the paper (Eqs. 1-15) after reading the pinned
official repository (no license file). The retrieval database is the stride-1 window
set of the training split, filled once by ``fit_database`` (called by the spec's
``training_setup``); it is stored as the training series plus its raw calendar marks,
and windows are formed as views on the fly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window

# Steps per day for each supported data frequency (the official frequency table).
STEPS_PER_DAY = {"h": 24, "t": 96, "15min": 96, "30min": 48, "d": 1, "w": 1, "m": 1, "s": 86400}


def window_stationarity(windows: torch.Tensor, n_sub: int) -> torch.Tensor:
    """Eq. (1): per-window stationarity score in ``[0, 1]`` for ``[N, L, C]`` windows.

    The window is cut into ``n_sub`` sub-windows (the last absorbs the remainder); ``v_mu``
    and ``v_sigma`` are the across-sub-window standard deviations of the per-channel
    sub-window means and standard deviations, averaged over channels, and both are
    normalised by the window's own per-channel standard deviation (averaged over channels).
    """
    length = windows.shape[1]
    size = length // n_sub
    if size < 2:
        raise ValueError("stationarity needs at least two steps per sub-window")
    means, stds = [], []
    for k in range(n_sub):
        stop = (k + 1) * size if k < n_sub - 1 else length
        part = windows[:, k * size:stop, :]
        means.append(part.mean(dim=1))
        stds.append(part.std(dim=1))
    v_mu = torch.stack(means, dim=1).std(dim=1).mean(dim=1)
    v_sigma = torch.stack(stds, dim=1).std(dim=1).mean(dim=1)
    scale = windows.std(dim=1).mean(dim=1) + 1e-8
    score = 0.5 * (1.0 - (v_mu / scale).clamp(0.0, 1.0)) + 0.5 * (1.0 - (v_sigma / scale).clamp(0.0, 1.0))
    return score.clamp(0.0, 1.0)


def _circular(diff: torch.Tensor, period: float) -> torch.Tensor:
    diff = diff.abs()
    return torch.minimum(diff, period - diff)


def time_aligned_bonus(
    query_mark: torch.Tensor, key_mark: torch.Tensor, seq_len: int, steps_per_day: int
) -> torch.Tensor:
    """Eq. (3): calendar alignment bonus ``B [B, N]`` rescaled by its maximum entry.

    ``query_mark [B, F]`` and ``key_mark [N, F]`` are raw marks at the window midpoint,
    laid out as ``[year, month, day, weekday, hour, minute]``. Hour (period 24), month
    (period 12) and minute (period 60) use exponential kernels on circular distances;
    weekday scores an exact match plus a weekday/weekend type match. The kernel widths
    and weights depend on whether the window is shorter than one day (official code).
    """
    bonus = query_mark.new_zeros(query_mark.shape[0], key_mark.shape[0])
    width = query_mark.shape[1]
    if width >= 5:
        hour = _circular(query_mark[:, 4:5] - key_mark[None, :, 4], 24.0)
        if seq_len < steps_per_day:
            bonus = bonus + 0.6 * torch.exp(-hour / 2.0)
        else:
            bonus = bonus + 0.3 * torch.exp(-hour / 4.0)
        q_day, k_day = query_mark[:, 3:4], key_mark[None, :, 3]
        bonus = bonus + 0.2 * (q_day == k_day).to(bonus.dtype)
        bonus = bonus + 0.1 * ((q_day >= 5) == (k_day >= 5)).to(bonus.dtype)
        month = _circular(query_mark[:, 1:2] - key_mark[None, :, 1], 12.0)
        if seq_len >= steps_per_day:
            bonus = bonus + 0.4 * torch.exp(-month / 2.0)
        else:
            bonus = bonus + 0.2 * torch.exp(-month / 3.0)
    if width >= 6 and seq_len < steps_per_day // 4:
        minute = _circular(query_mark[:, 5:6] - key_mark[None, :, 5], 60.0)
        bonus = bonus + 0.1 * torch.exp(-minute / 15.0)
    return bonus / bonus.max().clamp(min=1e-8)


def stochastic_mmr(
    candidate_sim: torch.Tensor,
    k: int,
    lam: float,
    temperature: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Eqs. (7)-(9): stochastic MMR over a ``[B, M]`` pool sorted by descending similarity.

    The first pick is the most similar candidate; each later pick is sampled from
    ``softmax(MMR / temperature)`` with ``MMR(i) = lam * S(i) - (1 - lam) * max_j Delta(i, j)``
    over the selected set and the proxy ``Delta(i, j) = 1 - |S(i) - S(j)|``. Candidates
    with a non-finite similarity (masked) are never sampled. Returns ``[B, M]`` booleans.
    """
    batch, pool = candidate_sim.shape
    finite = torch.isfinite(candidate_sim)
    sim = torch.where(finite, candidate_sim, torch.zeros_like(candidate_sim))
    redundancy_proxy = 1.0 - (sim[:, :, None] - sim[:, None, :]).abs()  # [B, M, M]
    selected = torch.zeros(batch, pool, dtype=torch.bool, device=candidate_sim.device)
    selected[:, 0] = finite[:, 0]
    rows = torch.arange(batch, device=candidate_sim.device)
    for _ in range(1, min(k, pool)):
        available = finite & ~selected
        has_choice = available.any(dim=1) & selected.any(dim=1)
        if not has_choice.any():
            break
        redundancy = redundancy_proxy.masked_fill(~selected[:, None, :], float("-inf")).amax(dim=2)
        score = (lam * sim - (1.0 - lam) * redundancy).masked_fill(~available, float("-inf"))
        safe = torch.where(has_choice[:, None], score, torch.zeros_like(score))
        probs = F.softmax(safe / temperature, dim=1)
        pick = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        selected[rows[has_choice], pick[has_choice]] = True
    return selected


class Model(nn.Module):
    """Linear forecaster plus a stationarity-aware retrieval branch (Fig. 2)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        top_k: int = 20,
        candidate_pool: int = 100,
        time_aware_weight: float = 0.5,
        sigma_min: float = 0.05,
        sigma_max: float = 0.3,
        lambda_min: float = 0.3,
        lambda_max: float = 0.9,
        mmr_temperature: float = 0.1,
        n_sub_windows: int = 6,
        freq: str = "h",
        eval_seed: int = 0,
    ) -> None:
        super().__init__()
        if seq_len // n_sub_windows < 2:
            raise ValueError("seq_len must give at least two steps per stationarity sub-window")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.top_k = top_k
        self.candidate_pool = candidate_pool
        self.time_aware_weight = time_aware_weight
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.mmr_temperature = mmr_temperature
        self.n_sub_windows = n_sub_windows
        self.steps_per_day = STEPS_PER_DAY[freq]
        self.eval_seed = eval_seed

        # Eq. (13): direct forecaster on the last-value-offset window, shared across channels.
        self.direct = nn.Linear(seq_len, pred_len)
        # Official code: a horizon linear map on the retrieved future before fusion.
        self.retrieval_proj = nn.Linear(pred_len, pred_len)
        # Eq. (15): final projection along the horizon.
        self.output_proj = nn.Linear(pred_len, pred_len)

        # Retrieval database (training split): series [T, C], raw marks [T, 6], and s-bar.
        self.register_buffer("db_series", torch.zeros(0, enc_in))
        self.register_buffer("db_marks", torch.zeros(0, 6))
        self.register_buffer("db_stationarity", torch.tensor(0.0))

    # ----------------------------------------------------------------- database
    @property
    def n_windows(self) -> int:
        return max(0, self.db_series.shape[0] - self.seq_len - self.pred_len + 1)

    @property
    def sigma(self) -> float:
        """Eq. (10): Gaussian bandwidth, larger for less stationary data."""
        s = float(self.db_stationarity)
        return self.sigma_min + (1.0 - s) * (self.sigma_max - self.sigma_min)

    @property
    def mmr_lambda(self) -> float:
        """Eq. (9): MMR balance, closer to plain Top-K for more stationary data."""
        s = float(self.db_stationarity)
        return self.lambda_min + s * (self.lambda_max - self.lambda_min)

    def _windows(self, start: int, stop: int) -> torch.Tensor:
        """Input windows ``[n, L, C]`` starting at ``start..stop-1`` (a view)."""
        span = self.db_series[start:stop + self.seq_len - 1]
        return span.unfold(0, self.seq_len, 1).transpose(1, 2)

    def _futures(self, index: torch.Tensor) -> torch.Tensor:
        """Paired futures ``[..., H, C]`` minus their own last value (official offset)."""
        steps = index[..., None] + self.seq_len + torch.arange(self.pred_len, device=index.device)
        future = self.db_series[steps]
        return future - future[..., -1:, :]

    def _chunk(self) -> int:
        return max(1, (1 << 22) // max(1, self.seq_len * self.enc_in))

    @torch.no_grad()
    def fit_database(self, series: torch.Tensor, marks: torch.Tensor | None = None) -> None:
        """Store the training series ``[T, C]`` (and raw marks ``[T, F]``) and Eq. (2) ``s-bar``."""
        if series.ndim != 2 or series.shape[1] != self.enc_in:
            raise ValueError(f"expected a [T, {self.enc_in}] training series, got {tuple(series.shape)}")
        if series.shape[0] < self.seq_len + self.pred_len:
            raise ValueError("training series is shorter than one input/future pair")
        device = self.direct.weight.device
        self.db_series = series.detach().to(device=device, dtype=torch.float32).clone()
        if marks is None:
            self.db_marks = torch.zeros(series.shape[0], 6, device=device)
        else:
            if marks.ndim != 2 or marks.shape[0] != series.shape[0]:
                raise ValueError("marks must be [T, F] on the series time axis")
            self.db_marks = marks.detach().to(device=device, dtype=torch.float32).clone()
        scores = []
        for start in range(0, self.n_windows, self._chunk()):
            stop = min(start + self._chunk(), self.n_windows)
            scores.append(window_stationarity(self._windows(start, stop), self.n_sub_windows))
        self.db_stationarity.fill_(torch.cat(scores).mean())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The database size depends on the training split; resize before the strict copy.
        for name in ("db_series", "db_marks"):
            key = prefix + name
            current = getattr(self, name)
            if key in state_dict and state_dict[key].shape != current.shape:
                setattr(self, name, current.new_empty(state_dict[key].shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ---------------------------------------------------------------- retrieval
    def temporal_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (4): Pearson similarity ``[B, N]`` of last-value-offset, flattened windows."""
        def centred(w: torch.Tensor) -> torch.Tensor:
            w = (w - w[:, -1:, :]).flatten(1)
            return F.normalize(w - w.mean(dim=1, keepdim=True), dim=1)

        query = centred(x)
        sims = []
        for start in range(0, self.n_windows, self._chunk()):
            stop = min(start + self._chunk(), self.n_windows)
            sims.append(query @ centred(self._windows(start, stop)).T)
        return torch.cat(sims, dim=1)

    def self_index(self, x: torch.Tensor) -> torch.Tensor:
        """Database index of each query window (``-1`` when it is not a database window)."""
        mid = self.seq_len // 2
        n = self.n_windows
        last = self.seq_len - 1
        keys = torch.cat(
            (self.db_series[:n], self.db_series[mid:mid + n], self.db_series[last:last + n]), dim=1
        )
        probe = torch.cat((x[:, 0], x[:, mid], x[:, -1]), dim=1)
        best, index = torch.cdist(probe, keys, p=1.0).min(dim=1)
        tolerance = 1e-5 * (1.0 + probe.abs().sum(dim=1))
        return torch.where(best <= tolerance, index, torch.full_like(index, -1))

    @torch.no_grad()
    def retrieve(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        """Eqs. (3)-(12): retrieval forecast ``[B, H, C]`` (zero before ``fit_database``)."""
        batch = x.shape[0]
        n = self.n_windows
        if n == 0:
            return x.new_zeros(batch, self.pred_len, self.enc_in)
        x = x.float()
        sim = self.temporal_similarity(x)
        if x_mark is not None and x_mark.ndim == 3 and self.time_aware_weight > 0:
            mid = self.seq_len // 2
            keys = self.db_marks[mid:mid + n]
            bonus = time_aligned_bonus(x_mark[:, mid].float(), keys, self.seq_len, self.steps_per_day)
            sim = (1.0 - self.time_aware_weight) * sim + self.time_aware_weight * bonus  # Eq. (5)
        if self.training:
            # Exclude database windows whose input or future span overlaps the query's.
            own = self.self_index(x)
            offset = (torch.arange(n, device=x.device)[None, :] - own[:, None]).abs()
            leak = (own[:, None] >= 0) & (offset <= self.seq_len + self.pred_len - 1)
            sim = sim.masked_fill(leak, float("-inf"))
        pool = min(self.candidate_pool, n)
        candidate_sim, candidate_idx = sim.topk(pool, dim=1)
        generator = None
        if not self.training:
            generator = torch.Generator(device=x.device).manual_seed(self.eval_seed)
        chosen = stochastic_mmr(candidate_sim, self.top_k, self.mmr_lambda, self.mmr_temperature, generator)
        # Eq. (11): Gaussian weights on d = 1 - S over the selected set.
        logits = -((1.0 - candidate_sim) ** 2) / (2.0 * self.sigma ** 2)
        kernel = torch.exp(logits.masked_fill(~chosen, float("-inf")))
        weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
        futures = self._futures(candidate_idx)  # [B, M, H, C]
        return torch.einsum("bm,bmhc->bhc", weights, futures)  # Eq. (12)

    # ------------------------------------------------------------------ forward
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        last = x_enc[:, -1:, :].detach()
        direct = self.direct((x_enc - last).transpose(1, 2)).transpose(1, 2)
        retrieved = self.retrieve(x_enc, x_mark_enc).to(x_enc.dtype)
        retrieved = self.retrieval_proj(retrieved.transpose(1, 2)).transpose(1, 2)
        fused = 0.5 * (direct + retrieved)  # Eq. (14)
        return self.output_proj(fused.transpose(1, 2)).transpose(1, 2) + last


def series_from_windows(dataset, pred_len: int):
    """Rebuild the ``[T, C]`` series and ``[T, F]`` marks behind a stride-1 window dataset.

    Items are ``(x, y, x_mark, y_mark, ...)`` with ``y`` ending in the ``pred_len`` future.
    """
    count = len(dataset)
    if count == 0:
        raise ValueError("empty training dataset")

    def as_float(value):
        return None if value is None else torch.as_tensor(value).float()

    def item(i):
        return tuple(as_float(value) for value in dataset[i][:4])

    first = item(0)
    if count > 1 and not torch.equal(first[0][1:], item(1)[0][:-1]):
        raise ValueError("SARAF needs a stride-1 sliding-window training dataset")
    rows, marks = [], []
    for i in range(count):
        x, _, x_mark, _ = item(i)
        rows.append(x[0])
        if x_mark is not None:
            marks.append(x_mark[0])
    x, y, x_mark, y_mark = item(count - 1)
    series = torch.cat((torch.stack(rows), x[1:], y[-pred_len:]))
    if len(marks) != count or x_mark is None or y_mark is None:
        return series, None
    return series, torch.cat((torch.stack(marks), x_mark[1:], y_mark[-pred_len:]))


__all__ = [
    "Model",
    "STEPS_PER_DAY",
    "series_from_windows",
    "stochastic_mmr",
    "time_aligned_bonus",
    "window_stationarity",
]

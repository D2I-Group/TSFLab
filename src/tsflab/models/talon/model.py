"""Local TALON implementation (GPT-2 variant) from the paper and the pinned official code.

TALON (arXiv 2508.07195) cuts each channel's history into non-overlapping
segments, describes every segment by three statistics (STL trend strength,
local variation, lag-1 autocorrelation), and routes it through a Heterogeneous
Temporal Encoder: a noisy top-k gate whose noise scale is driven by those
statistics mixes a Linear, a CNN and an LSTM expert, each mapping the segment to
one GPT-2 token. A frozen GPT-2 trunk models the token sequence and an MLP maps
every output position to the next segment; inference rolls the next-segment
forecast forward autoregressively.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.gpt2_backbone import GPT2Backbone, GPT2Config
from tsflab.models._components.revin import RevIN
from tsflab.models._components.segment_mlp import SegmentMLP

NUM_EXPERTS = 3  # Linear, CNN, LSTM (Sec. IV-B)


# ---------------------------------------------------------------------------
# STL (Cleveland et al., 1990) as a linear operator
# ---------------------------------------------------------------------------
# Without robustness weights every STL smoother is a fixed weighted average of
# the input, so the seasonal and trend parts are linear maps of the series. The
# maps below follow the classic STL procedure (LOESS degree 1, tricube weights,
# all jumps 1, cycle-subseries smoothing with one-step extrapolation at both
# ends, low-pass filter MA(p) -> MA(p) -> MA(3) -> LOESS) with the default
# windows of the official dependency (statsmodels STL: seasonal 7, trend the
# smallest odd integer >= 1.5 p / (1 - 1.5 / seasonal), low-pass the smallest odd
# integer > p, 5 inner iterations and no outer iterations).
def _loess_row(n: int, window: int, degree: int, xs: int, left: int, right: int) -> np.ndarray | None:
    """Weights of the local fit at 1-based position ``xs`` over points ``left..right``."""
    h = float(max(xs - left, right - xs))
    if window > n:
        h += (window - n) // 2
    positions = np.arange(left, right + 1, dtype=np.float64)
    distance = np.abs(positions - xs)
    weights = np.where(distance <= 0.001 * h, 1.0, (1.0 - (distance / h) ** 3) ** 3) if h > 0 \
        else np.ones_like(distance)
    weights = np.where(distance <= 0.999 * h, weights, 0.0)
    total = weights.sum()
    if total <= 0:
        return None
    weights = weights / total
    if h > 0 and degree > 0:
        center = (weights * positions).sum()
        spread = (weights * (positions - center) ** 2).sum()
        if math.sqrt(spread) > 0.001 * (n - 1):
            weights = weights * ((xs - center) / spread * (positions - center) + 1.0)
    row = np.zeros(n)
    row[left - 1:right] = weights
    return row


def _loess_matrix(n: int, window: int, degree: int = 1) -> np.ndarray:
    """``[n, n]`` LOESS smoother evaluated at every point (jump 1)."""
    if n < 2:
        return np.eye(n)
    matrix = np.zeros((n, n))
    left, right = 1, min(window, n)
    half = (window + 2) // 2
    for i in range(1, n + 1):
        if window < n and i > half and right != n:
            left, right = left + 1, right + 1
        row = _loess_row(n, window, degree, i, left, right)
        matrix[i - 1] = row if row is not None else np.eye(n)[i - 1]
    return matrix


def _moving_average(n: int, length: int) -> np.ndarray:
    """``[n - length + 1, n]`` running mean of ``length`` points."""
    out = np.zeros((n - length + 1, n))
    for i in range(n - length + 1):
        out[i, i:i + length] = 1.0 / length
    return out


@cache
def stl_operators(n: int, period: int, seasonal: int = 7, inner_iter: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(S, T)`` ``[n, n]`` with ``seasonal = S y`` and ``trend = T y`` (non-robust STL)."""
    if period < 2 or seasonal < 3 or seasonal % 2 == 0:
        raise ValueError("STL needs period >= 2 and an odd seasonal window >= 3")
    trend_window = math.ceil(1.5 * period / (1 - 1.5 / seasonal))
    trend_window += trend_window % 2 == 0
    low_pass_window = period + 1
    low_pass_window += low_pass_window % 2 == 0
    # Cycle-subseries smoothing with extrapolation to one point before and after.
    cycle = np.zeros((n + 2 * period, n))
    for j in range(period):
        count = (n - (j + 1)) // period + 1
        index = np.arange(count) * period + j
        inner = _loess_matrix(count, seasonal)
        rows = np.zeros((count + 2, count))
        rows[1:count + 1] = inner
        first = _loess_row(count, seasonal, 1, 0, 1, min(seasonal, count))
        rows[0] = first if first is not None else rows[1]
        last = _loess_row(count, seasonal, 1, count + 1, max(1, count - seasonal + 1), count)
        rows[count + 1] = last if last is not None else rows[count]
        for m in range(count + 2):
            cycle[m * period + j, index] = rows[m]
    low_pass = (
        _loess_matrix(n, low_pass_window)
        @ _moving_average(n + 2, 3)
        @ _moving_average(n + period + 1, period)
        @ _moving_average(n + 2 * period, period)
    )
    trend_smoother = _loess_matrix(n, trend_window)
    identity = np.eye(n)
    trend = np.zeros((n, n))
    season = np.zeros((n, n))
    for _ in range(inner_iter):
        smoothed = cycle @ (identity - trend)
        season = smoothed[period:period + n] - low_pass @ smoothed
        trend = trend_smoother @ (identity - season)
    return season, trend


# ---------------------------------------------------------------------------
# Pattern quantification (Sec. IV-B, Appendix D, Algorithm 1)
# ---------------------------------------------------------------------------
class PatchComplexity(nn.Module):
    """Three descriptors per segment ``[M, S] -> [M, 3]``: trend strength, local
    variation, |lag-1 autocorrelation|, each rounded to three decimals."""

    def __init__(self, token_len: int) -> None:
        super().__init__()
        season, trend = stl_operators(token_len, max(token_len // 2, 2))
        self.register_buffer("season_op", torch.from_numpy(season.copy()), persistent=False)
        self.register_buffer("trend_op", torch.from_numpy(trend.copy()), persistent=False)

    @torch.no_grad()
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        p = patches.double()
        season = p @ self.season_op.double().T
        trend = p @ self.trend_op.double().T
        resid = p - season - trend
        var_deseason = (p - season).var(dim=-1, correction=0)
        var_resid = resid.var(dim=-1, correction=0)
        # A constant segment has an exactly zero deseasonalized variance under STL
        # (official ``var_deseasonal == 0`` branch); the linear operators leave
        # rounding noise there, so decide constancy on the segment itself.
        nonconstant = p.std(dim=-1, correction=0) > 0
        valid = nonconstant & (var_deseason > 0)
        safe = torch.where(valid, var_deseason, torch.ones_like(var_deseason))
        strength = torch.where(valid, 1 - var_resid / safe, torch.zeros_like(safe))
        diff_std = p.diff(dim=-1).std(dim=-1, correction=0)
        variation = torch.sigmoid(torch.log1p(diff_std) - 1.0)
        centered = p - p.mean(dim=-1, keepdim=True)
        energy = centered.square().sum(dim=-1)
        lag1 = (centered[..., :-1] * centered[..., 1:]).sum(dim=-1)
        autocorr = torch.where(nonconstant, (lag1 / torch.where(nonconstant, energy, torch.ones_like(energy))).abs(),
                               torch.zeros_like(energy))
        features = torch.stack([strength, variation, autocorr], dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return (torch.round(features * 1000) / 1000).to(patches.dtype)


# ---------------------------------------------------------------------------
# Heterogeneous Temporal Encoder
# ---------------------------------------------------------------------------
class LinearExpert(nn.Module):
    """Eq. (7): ``s W_Linear``."""

    def __init__(self, token_len: int, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(token_len, width)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(s)


class CNNExpert(nn.Module):
    """Eq. (8): ``W_proj Conv2(ReLU(Conv1(s)))`` with a width-3 then a 1x1 convolution."""

    def __init__(self, token_len: int, hidden: int, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, 1, kernel_size=1)
        self.proj = nn.Linear(token_len, width)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv2(F.relu(self.conv1(s.unsqueeze(1)))).squeeze(1))


class LSTMExpert(nn.Module):
    """Eq. (9): ``W_proj LSTM(s)[-1]`` over the segment's time steps."""

    def __init__(self, hidden: int, width: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden, width)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(s.unsqueeze(-1))
        return self.proj(out[:, -1])


def cv_squared(values: torch.Tensor) -> torch.Tensor:
    """Squared coefficient of variation ``var / (mean^2 + 1e-10)`` (unbiased variance)."""
    if values.numel() <= 1:
        return values.new_zeros(())
    values = values.float()
    return values.var() / (values.mean() ** 2 + 1e-10)


class HeterogeneousTemporalEncoder(nn.Module):
    """Pattern-adaptive routing (Eqs. 1-6) over Linear/CNN/LSTM experts (Eqs. 7-10).

    ``forward(s [M, S], c [M, 3]) -> (e [M, width], L_MoE)``. In training mode the
    routing scores get Gaussian noise with scale ``Softplus(c~) + noise_epsilon``.
    """

    def __init__(self, token_len: int, width: int, hidden: int, top_k: int,
                 noise_epsilon: float = 1e-2) -> None:
        super().__init__()
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        # Eq. (1): z~ = ReLU(s W0t) W1t ; Eq. (2): c~ = ReLU(c W0c) W1c (no biases).
        self.segment_gate = nn.Sequential(nn.Linear(token_len, hidden, bias=False), nn.ReLU(),
                                          nn.Linear(hidden, NUM_EXPERTS, bias=False))
        self.complexity_gate = nn.Sequential(nn.Linear(3, hidden, bias=False), nn.ReLU(),
                                             nn.Linear(hidden, NUM_EXPERTS, bias=False))
        self.projection = nn.Parameter(torch.eye(NUM_EXPERTS))  # W_H, Eq. (4)
        self.experts = nn.ModuleList([
            LinearExpert(token_len, width),
            CNNExpert(token_len, hidden, width),
            LSTMExpert(hidden, width),
        ])

    def route(self, s: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse gates ``[M, K]`` (Eqs. 3-6) and the per-expert load ``[K]``."""
        clean = self.segment_gate(s)
        noisy = self.training
        if noisy:
            scale = F.softplus(self.complexity_gate(c)) + self.noise_epsilon
            latent = clean + torch.randn_like(clean) * scale  # Eq. (3)
        else:
            latent = clean
        scores = latent @ self.projection  # Eq. (4)
        probs = torch.softmax(scores, dim=-1)
        top_values, top_index = probs.topk(min(self.top_k + 1, NUM_EXPERTS), dim=-1)
        kept = top_values[:, : self.top_k]
        # Eqs. (5)-(6): softmax restricted to the top-k scores.
        kept = kept / (kept.sum(dim=-1, keepdim=True) + 1e-6)
        gates = torch.zeros_like(probs).scatter(1, top_index[:, : self.top_k], kept)
        if noisy and self.top_k < NUM_EXPERTS:
            load = self._smooth_load(clean @ self.projection, scores, scale, top_index)
        else:
            load = (gates > 0).sum(dim=0).to(gates.dtype)
        return gates, load

    def _smooth_load(self, clean_scores, noisy_scores, scale, top_index) -> torch.Tensor:
        """Differentiable load: ``sum_m P(expert j in top-k)`` under the routing noise.

        The latent noise ``eps * scale`` is propagated through ``W_H``: score ``j``
        has standard deviation ``sqrt(sum_i (scale_i W_H[i, j])^2)``.
        """
        std = torch.sqrt((scale.unsqueeze(-1) * self.projection.unsqueeze(0)).square().sum(dim=1)) + 1e-12
        ranked = noisy_scores.gather(1, top_index)
        threshold_in = ranked[:, self.top_k: self.top_k + 1]  # (k+1)-th score
        threshold_out = ranked[:, self.top_k - 1: self.top_k]  # k-th score
        is_in = noisy_scores > threshold_in
        normal = torch.distributions.Normal(clean_scores.new_zeros(()), clean_scores.new_ones(()))
        prob_in = normal.cdf((clean_scores - threshold_in) / std)
        prob_out = normal.cdf((clean_scores - threshold_out) / std)
        return torch.where(is_in, prob_in, prob_out).sum(dim=0)

    def forward(self, s: torch.Tensor, c: torch.Tensor, loss_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
        gates, load = self.route(s, c)
        balance = loss_weight * (cv_squared(gates.sum(dim=0)) + cv_squared(load))  # Eq. (11)
        out = s.new_zeros(s.shape[0], self.experts[0].proj.out_features)
        for j, expert in enumerate(self.experts):
            rows = torch.nonzero(gates[:, j] > 0, as_tuple=True)[0]
            if rows.numel():
                out = out.index_add(0, rows, gates[rows, j: j + 1] * expert(s[rows]))  # Eq. (10)
        return out, balance


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """TALON with a frozen GPT-2 trunk; ``forward`` is the rolling forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        token_len: int = 96,
        top_k: int = 3,
        expert_hidden: int = 1024,
        mlp_hidden_dim: int = 1024,
        mlp_hidden_layers: int = 2,
        mlp_activation: str = "tanh",
        dropout: float = 0.1,
        llm_layers: int = 6,
        moe_weight: float = 0.02,
    ) -> None:
        super().__init__()
        if seq_len % token_len:
            raise ValueError("seq_len must be a multiple of token_len (context = N segments)")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.token_len = token_len
        self.token_num = seq_len // token_len
        self.moe_weight = moe_weight
        self.llm = GPT2Backbone(GPT2Config(n_layer=llm_layers))
        if self.token_num > self.llm.config.n_positions:
            raise ValueError("seq_len / token_len exceeds the LLM context")
        for parameter in self.llm.parameters():  # the LLM stays frozen
            parameter.requires_grad_(False)
        width = self.llm.config.n_embd
        self.normalizer = RevIN(1, affine=False)
        self.complexity = PatchComplexity(token_len)
        self.encoder = HeterogeneousTemporalEncoder(token_len, width, expert_hidden, top_k)
        self.decoder = SegmentMLP(width, token_len, mlp_hidden_dim, mlp_hidden_layers,
                                  dropout, mlp_activation)

    def next_segments(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map segments ``s_1..s_N`` of ``[B, N*S, C]`` to predictions of ``s_2..s_{N+1}``.

        Returns the predictions (input scale) and the expert-balance loss
        ``alpha * L_MoE`` of this pass.
        """
        if x_enc.ndim != 3 or x_enc.shape[1] % self.token_len:
            raise ValueError(f"TALON expects [batch, N * {self.token_len}, channels]")
        batch, length, channels = x_enc.shape
        series = x_enc.permute(0, 2, 1).reshape(batch * channels, length, 1)  # channel independence
        raw_segments = series.squeeze(-1).unfold(-1, self.token_len, self.token_len)
        # Routing cues are computed on the (dataset-scaled) input before instance normalization.
        cues = self.complexity(raw_segments.reshape(-1, self.token_len))
        normalized = self.normalizer(series, "norm")
        segments = normalized.squeeze(-1).unfold(-1, self.token_len, self.token_len)
        tokens, balance = self.encoder(segments.reshape(-1, self.token_len), cues, self.moe_weight)
        hidden = self.llm(tokens.view(batch * channels, -1, tokens.shape[-1]))  # Eq. (13)
        predicted = self.decoder(hidden).reshape(batch * channels, length, 1)
        predicted = self.normalizer(predicted, "denorm")
        return predicted.reshape(batch, channels, length).permute(0, 2, 1), balance

    def training_loss(self, x_enc: torch.Tensor, future: torch.Tensor, criterion) -> torch.Tensor:
        """Next-segment loss on the history shifted by one segment plus ``alpha * L_MoE`` (Eq. 14
        without the alignment term)."""
        available = min(self.token_len, future.shape[1])
        target = torch.cat([x_enc, future[:, :available]], dim=1)[:, self.token_len:]
        predicted, balance = self.next_segments(x_enc)
        return criterion(predicted[:, : target.shape[1]], target) + balance

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        """Autoregressive rolling forecast of ``pred_len`` steps."""
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"TALON expects [batch, {self.seq_len}, channels]")
        window, outputs = x_enc, []
        for _ in range(math.ceil(self.pred_len / self.token_len)):
            predicted, _ = self.next_segments(window)
            segment = predicted[:, -self.token_len:]
            outputs.append(segment)
            # Slide the context; the routing cues are recomputed for the new window.
            window = torch.cat([window[:, self.token_len:], segment], dim=1)
        return torch.cat(outputs, dim=1)[:, : self.pred_len]

"""MFRS: multi-frequency reference series with variate-to-reference cross-attention (Yu et al., 2025).

Before training, the Base-Pattern Extractor reads the amplitude spectrum of the
training split: primary base patterns (integer periods) come from the spectrum
re-indexed by period (Sec. 3.2.2, Algorithm 1) and the top-``Q`` harmonics of
those periods by a cross-channel amplitude-ratio score (Algorithm 2). The
Reference-Series Generator turns every pattern into one reference series
(sine, sawtooth, rectangle or pulse; Sec. 3.2.3, App. B) whose phase follows the
absolute time step of the window (Sec. 3.3); when a window carries no
timestamps, the step is recovered by correlating it against a stored stretch
of the training series (Algorithm 3). The forecaster is an inverted Transformer
in which each variate token attends only to the reference-series tokens
(Sec. 3.4), so variates never attend to one another (App. A).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import elapsed_minutes as stamp_minutes
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._inputs import check_window

WAVEFORMS = ("sine", "sawtooth", "rectangle", "pulse")


# ------------------------------------------------------------------ timestamps


def informative_marks(marks: torch.Tensor | None, length: int) -> bool:
    """Marks carry real dates (a year is set) and at least two steps exist."""
    return (
        marks is not None
        and marks.ndim == 3
        and marks.shape[-1] == 6
        and length > 1
        and bool((marks[..., 0] > 0).all())
    )


# ------------------------------------------------------------- base patterns


def amplitude_spectrum(series: torch.Tensor) -> torch.Tensor:
    """``[L, C] -> [L // 2 + 1, C]`` amplitude of the rFFT over time (phase discarded, Sec. 3.2.1)."""
    series = series.double()
    return torch.fft.rfft(series - series.mean(0, keepdim=True), dim=0).abs()


def amplitude_at(spectrum: torch.Tensor, length: int, period: int, harmonic: int = 1) -> torch.Tensor:
    """``Phi(harmonic / period)``: the rFFT bin nearest to that frequency, zero outside ``(0, 1/2]``."""
    index = int(round(length * harmonic / period))
    if index < 1 or index > length // 2:
        return torch.zeros_like(spectrum[0])
    return spectrum[index]


def period_spectrum(spectrum: torch.Tensor, length: int, max_period: int) -> torch.Tensor:
    """``Psi(T)`` for ``T = 0..Lp`` from a ``[L // 2 + 1]`` spectrum (Sec. 3.2.2, Fig. 2b).

    Every rFFT bin ``l >= 1`` is assigned to its nearest integer period
    ``round(L / l)``; ``Psi(T)`` is the largest amplitude assigned to ``T`` (zero
    when no bin falls on ``T``). ``Psi(0) = Psi(1) = 0``: period 1 is frequency 1,
    beyond the Nyquist frequency.
    """
    limit = min(max_period, length)
    index = torch.arange(1, length // 2 + 1)
    period = torch.round(length / index.double()).long()
    keep = (period >= 2) & (period <= limit)
    psi = torch.zeros(limit + 1, dtype=torch.float64)
    return psi.scatter_reduce(0, period[keep], spectrum[index[keep]].double(), reduce="amax")


def primary_base_patterns(spectrum: torch.Tensor, length: int, max_period: int) -> list[int]:
    """Algorithm 1: ``T`` is a primary base pattern when ``Psi(T)`` is the largest of
    ``Psi(1..2T)``; that interval is then zeroed. ``T`` runs from 2 to ``Lp / 2``."""
    psi = period_spectrum(spectrum, length, max_period)
    found = []
    for period in range(2, (psi.shape[0] - 1) // 2 + 1):
        window = psi[1 : 2 * period + 1]
        if window.max() > 0 and int(window.argmax()) + 1 == period:
            found.append(period)
            psi[1 : 2 * period + 1] = 0.0
    return found


def harmonic_base_patterns(
    spectra: torch.Tensor, length: int, primaries: list[int], count: int
) -> list[tuple[int, int]]:
    """Algorithm 2: score harmonics ``k f_P^m`` by ``sum_c Phi_c(k f) / Phi_c(f)``, keep the top ``count``.

    With periods ascending (``T_1 < ... < T_M``), ``k`` runs to ``floor(T_1 / 2)``
    for the first pattern and to ``floor(T_m / (2 T_{m-1}))`` afterwards, so a
    harmonic stays below half the frequency of the next shorter pattern.
    Returns ``(period, harmonic)`` pairs.
    """
    periods = sorted(set(primaries))
    scores: dict[tuple[int, int], float] = {}
    for position, period in enumerate(periods):
        top = period // 2 if position == 0 else period // (2 * periods[position - 1])
        base = amplitude_at(spectra, length, period)
        for harmonic in range(2, top + 1):
            value = amplitude_at(spectra, length, period, harmonic)
            ratio = torch.where(base > 0, value / base.clamp_min(1e-300), torch.zeros_like(base))
            scores[(period, harmonic)] = float(ratio.sum())
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    return ranked[:count]


def reference_waveform(
    steps: torch.Tensor, period: torch.Tensor, harmonic: torch.Tensor, waveform: str
) -> torch.Tensor:
    """Appendix B for integer steps ``t``: one reference value per pattern ``(T, k)``.

    ``phase = (t * k) mod T``; sine ``sin(2 pi phase / T)``, sawtooth ``phase``,
    rectangle ``floor(2 k t / T) mod 2 = floor(2 phase / T)``, pulse ``[phase = 0]``.
    """
    phase = torch.remainder(steps * harmonic, period)
    if waveform == "sine":
        return torch.sin(2 * math.pi * phase.double() / period.double()).float()
    if waveform == "sawtooth":
        return phase.float()
    if waveform == "rectangle":
        return torch.div(2 * phase, period, rounding_mode="floor").float()
    if waveform == "pulse":
        return (phase == 0).float()
    raise ValueError(f"waveform must be one of {WAVEFORMS}")


def _reduced(period: int, harmonic: int) -> tuple[int, int]:
    divisor = math.gcd(period, harmonic)
    return period // divisor, harmonic // divisor


# --------------------------------------------------------------- forecaster


class ReferenceCrossAttentionLayer(nn.Module):
    """Post-norm block: variate queries attend to reference keys/values, then a GELU FFN (Sec. 3.4)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.attention = AttentionLayer(FullAttention(mask_flag=False, attention_dropout=dropout), d_model, n_heads)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, variates: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(variates, references, references, None)
        hidden = self.norm1(variates + self.dropout(attended))
        update = self.dropout(self.ff2(self.dropout(self.activation(self.ff1(hidden)))))
        return self.norm2(hidden + update)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        waveform: str = "sine",
        periods: tuple[int, ...] | list[int] = (),
        extract: bool = True,
        max_period: int = 1000,
        harmonics: int = 8,
        share_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.waveform = waveform
        self.periods = tuple(int(period) for period in periods)
        self.extract = extract
        self.max_period = max_period
        self.harmonics = harmonics
        initial = self.periods or (seq_len,)  # placeholder until training_setup fits the patterns
        self.register_buffer("pattern_period", torch.tensor(initial, dtype=torch.long))
        self.register_buffer("pattern_harmonic", torch.ones(len(initial), dtype=torch.long))
        self.register_buffer("align_series", torch.zeros(0, enc_in))
        self.register_buffer("align_start", torch.zeros((), dtype=torch.long))
        self.embedding = nn.Linear(seq_len, d_model)
        # Sec. 3.4: variates and reference series share the embedding unless told otherwise.
        self.reference_embedding = None if share_embedding else nn.Linear(seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            ReferenceCrossAttentionLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, pred_len)

    # -- base-pattern extraction and alignment ------------------------------------

    @torch.no_grad()
    def fit_reference_series(self, series: torch.Tensor) -> None:
        """Run the extractor on the ``[L, C]`` training series and store the alignment stretch."""
        series = series.detach().double().cpu()
        if series.ndim != 2 or series.shape[1] != self.enc_in or series.shape[0] < 4:
            raise ValueError(f"MFRS needs a [time, {self.enc_in}] training series of at least 4 steps")
        length = series.shape[0]
        chosen = self._base_patterns(series)
        device = self.pattern_period.device
        self.pattern_period = torch.tensor([p for p, _ in chosen], dtype=torch.long, device=device)
        self.pattern_harmonic = torch.tensor([k for _, k in chosen], dtype=torch.long, device=device)
        span = min(length, int(self.pattern_period.max()) + self.seq_len)
        self.align_series = series[length - span :].float().to(device)
        self.align_start = torch.tensor(length - span, dtype=torch.long, device=device)

    def _base_patterns(self, series: torch.Tensor) -> list[tuple[int, int]]:
        """Extracted primary and harmonic patterns, then the manual periods, one per reduced pattern."""
        length = series.shape[0]
        patterns: list[tuple[int, int]] = [(period, 1) for period in self.periods]
        if self.extract:
            spectra = amplitude_spectrum(series)
            primaries = primary_base_patterns(spectra.mean(1), length, self.max_period)
            patterns = [(period, 1) for period in primaries] + patterns
            patterns += harmonic_base_patterns(spectra, length, primaries, self.harmonics)
        unique: dict[tuple[int, int], tuple[int, int]] = {}
        for period, harmonic in patterns:
            unique.setdefault(_reduced(period, harmonic), (period, harmonic))
        return list(unique.values()) or [(self.seq_len, 1)]

    def align_steps(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Algorithm 3: start step maximizing the summed per-channel Pearson correlation."""
        batch = x_enc.shape[0]
        if self.align_series.shape[0] < self.seq_len:
            return torch.zeros(batch, dtype=torch.long, device=x_enc.device)
        windows = self.align_series.unfold(0, self.seq_len, 1)  # [candidates, C, S]
        windows = windows - windows.mean(-1, keepdim=True)
        windows = windows / windows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        observed = x_enc.transpose(1, 2).to(windows.dtype)
        observed = observed - observed.mean(-1, keepdim=True)
        observed = observed / observed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        score = torch.einsum("bcs,tcs->bt", observed, windows)
        return self.align_start + score.argmax(1)

    def start_steps(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """Absolute step of each window's first point (Sec. 3.3)."""
        if informative_marks(x_mark_enc, self.seq_len):
            minutes = stamp_minutes(x_mark_enc[:, [0, -1]])
            interval = torch.div(minutes[:, 1] - minutes[:, 0], self.seq_len - 1, rounding_mode="floor")
            if bool((interval > 0).all()):
                return torch.div(minutes[:, 0], interval, rounding_mode="floor").to(x_enc.device)
        return self.align_steps(x_enc)

    def reference_series(self, start: torch.Tensor) -> torch.Tensor:
        """``[B] -> [B, N, S]``: every pattern's reference series over the window's steps."""
        steps = start[:, None, None] + torch.arange(self.seq_len, device=start.device)[None, None, :]
        return reference_waveform(
            steps, self.pattern_period[None, :, None], self.pattern_harmonic[None, :, None], self.waveform
        )

    # -- forecasting ---------------------------------------------------------------

    @staticmethod
    def standardize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token DC blocking over time: ``(v - mean) / sqrt(var + 1e-5)`` on the last axis."""
        mean = values.mean(-1, keepdim=True).detach()
        scale = values.var(-1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
        return (values - mean) / scale, mean, scale

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        variates, mean, scale = self.standardize(x_enc.transpose(1, 2))
        references, _, _ = self.standardize(self.reference_series(self.start_steps(x_enc, x_mark_enc)))
        tokens = self.embedding_dropout(self.embedding(variates))
        embed = self.embedding if self.reference_embedding is None else self.reference_embedding
        keys = self.embedding_dropout(embed(references.to(tokens.dtype)))
        for layer in self.layers:
            tokens = layer(tokens, keys)
        forecast = self.projection(self.norm(tokens)) * scale + mean  # invert-norm
        return forecast.transpose(1, 2)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The fitted patterns and alignment stretch have data-dependent sizes.
        for name in ("pattern_period", "pattern_harmonic", "align_series"):
            key = prefix + name
            buffer = self._buffers[name]
            if key in state_dict and state_dict[key].shape != buffer.shape:
                self._buffers[name] = torch.empty(state_dict[key].shape, dtype=buffer.dtype, device=buffer.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


__all__ = [
    "Model",
    "amplitude_spectrum",
    "harmonic_base_patterns",
    "period_spectrum",
    "primary_base_patterns",
    "reference_waveform",
    "stamp_minutes",
]

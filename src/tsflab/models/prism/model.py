"""Local PRISM: a binary time tree of overlapping segments with routed Haar bands.

Paper map (Chen et al., arXiv 2512.24898, Section 3.2 and Fig. 1): per-series
z-normalization; recursive bisection of the context into two overlapping halves
(the time hierarchy); a same-length Haar band decomposition of every segment (the
frequency hierarchy, K bands); a band router that scores six summary statistics of
each band with a two-layer MLP and turns the scores into weights with a
temperature softmax; the band-weighted sum of each half feeds the next level;
per-band two-layer MLP heads map every routed band to the horizon, a
squeeze-and-excitation gate reweights the band forecasts, and their sum is
de-normalized.

The pinned official code (nerdslab/PRISM, ``src/arch/prism.py``) fixes the
details the paper leaves open; see the model card for the full mapping.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants of the official tree (not exposed as command-line options there).
ROUTER_TEMPERATURE = 0.2
FORECAST_HIDDEN = 128
SELECTOR_HIDDEN = 32
NORM_EPS = 1e-5
FEATURE_EPS = 1e-6
# Exponential-moving-average ladder used when a segment is too short for K - 1
# Haar levels (time constants tau_i = 8 * 3**i).
EMA_TAU0 = 8.0
EMA_GROW = 3.0


def split_length(length: int, overlap: int) -> int:
    """Length of each half when ``length`` is bisected with ``overlap`` extra samples.

    Both halves extend ``ov = min(overlap, length // 2, length - length // 2)`` samples
    past the midpoint; for odd lengths the right half is trimmed to the left half's
    length so the two segments have equal size ``length // 2 + ov``.
    """
    mid = length // 2
    ov = min(overlap, mid, length - mid)
    return mid + ov if ov > 0 else mid


def split_halves(x: torch.Tensor, overlap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlapping left ``x[..., :n]`` and right ``x[..., L - n:]`` halves, ``n = split_length``."""
    length = x.shape[-1]
    size = split_length(length, overlap)
    if 2 * size < length:
        raise ValueError("a segment without overlap must have even length")
    return x[..., :size], x[..., length - size:]


def haar_bands(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Same-length Haar band decomposition ``[B, C, L] -> [B, C, levels + 1, L]``.

    Analysis runs ``levels`` orthonormal Haar steps on the running approximation
    (``lo = (a_{2n} + a_{2n+1}) / sqrt 2``, ``hi = (a_{2n+1} - a_{2n}) / sqrt 2``); an
    odd approximation is first extended on the right by mirror reflection. Each
    detail and the final approximation are then synthesized back to length ``L`` on
    their own (zero for every other coefficient), so band 0 is the finest detail,
    band ``levels`` the coarsest approximation, and the bands sum exactly to ``x``.
    """
    scale = 2.0 ** -0.5
    lengths = [x.shape[-1]]
    details = []
    approx = x
    for _ in range(levels):
        if approx.shape[-1] % 2:
            approx = F.pad(approx, (0, 1), mode="reflect")
        even, odd = approx[..., 0::2], approx[..., 1::2]
        details.append((odd - even) * scale)
        approx = (even + odd) * scale
        lengths.append(approx.shape[-1])

    def upsample(coeff: torch.Tensor, high: bool, target: int) -> torch.Tensor:
        first = -coeff if high else coeff
        return (torch.stack((first, coeff), dim=-1).flatten(-2) * scale)[..., :target]

    bands = []
    for level, detail in enumerate(details):
        band = upsample(detail, True, lengths[level])
        for finer in range(level - 1, -1, -1):
            band = upsample(band, False, lengths[finer])
        bands.append(band)
    band = approx
    for finer in range(levels - 1, -1, -1):
        band = upsample(band, False, lengths[finer])
    bands.append(band)
    return torch.stack(bands, dim=2)


def ema_ladder_bands(x: torch.Tensor, bands: int) -> torch.Tensor:
    """Cascaded EMA low-pass ladder ``[B, C, L] -> [B, C, bands, L]`` (fine to coarse).

    ``s_i`` is the EMA (``y_t = a x_t + (1 - a) y_{t-1}``, ``y_0 = x_0``) of
    ``s_{i-1}`` with ``a = 1 - exp(-1 / tau_i)``; the bands are ``x - s_1``,
    ``s_{i-1} - s_i`` and ``s_{K-1}``, which sum to ``x``.
    """
    lows = []
    previous = x
    for index in range(bands - 1):
        alpha = 1.0 - math.exp(-1.0 / (EMA_TAU0 * EMA_GROW**index))
        steps = [previous[..., 0]]
        for t in range(1, previous.shape[-1]):
            steps.append(alpha * previous[..., t] + (1.0 - alpha) * steps[-1])
        previous = torch.stack(steps, dim=-1)
        lows.append(previous)
    out = [x - lows[0]]
    out.extend(lows[i - 1] - lows[i] for i in range(1, len(lows)))
    out.append(lows[-1])
    return torch.stack(out, dim=2)


def decompose(x: torch.Tensor, bands: int) -> torch.Tensor:
    """Fixed (non-differentiated) band decomposition of one segment into ``bands`` bands.

    Haar with ``bands - 1`` levels when ``bands - 1 <= floor(log2 L)``; otherwise the
    EMA ladder, as in the official dispatcher.
    """
    with torch.no_grad():
        if bands - 1 > int(math.log2(x.shape[-1])):
            return ema_ladder_bands(x, bands)
        return haar_bands(x, bands - 1)


def band_statistics(bands: torch.Tensor) -> torch.Tensor:
    """Six router inputs per band ``[B, C, K, L] -> [B, C, K, 6]`` (Section 3.2).

    ``(mean, std, max |b|, mean |diff b|, mean |diff^2 b|, max |b| / (std + eps))``
    with the population standard deviation; computed without gradient.
    """
    with torch.no_grad():
        mean = bands.mean(dim=-1)
        std = bands.std(dim=-1, unbiased=False)
        peak = bands.abs().amax(dim=-1)
        first = torch.diff(bands, dim=-1)
        second = torch.diff(first, dim=-1)
        return torch.stack(
            (
                mean,
                std,
                peak,
                first.abs().mean(dim=-1),
                second.abs().mean(dim=-1),
                peak / (std + FEATURE_EPS),
            ),
            dim=-1,
        )


class BandRouter(nn.Module):
    """Importance weights of the 2K bands of one node's two halves.

    A two-layer LeakyReLU MLP scores each band's statistics; the scores of each half
    pass a temperature softmax over its K bands, and the resulting band weights of
    the left and right halves then compete in a second temperature softmax over the
    two halves (official ``TimeTree.forward``). One MLP serves both halves.
    """

    def __init__(self, hidden: int = SELECTOR_HIDDEN, temperature: float = ROUTER_TEMPERATURE) -> None:
        super().__init__()
        self.temperature = temperature
        self.score = nn.Sequential(nn.Linear(6, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1))

    def band_weights(self, bands: torch.Tensor) -> torch.Tensor:
        logits = self.score(band_statistics(bands)).squeeze(-1)
        return torch.softmax(logits / self.temperature, dim=-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.stack((self.band_weights(left), self.band_weights(right)), dim=-1)
        pair = torch.softmax(pair / self.temperature, dim=-1)
        return pair[..., 0], pair[..., 1]


class SqueezeExcitationGate(nn.Module):
    """Band gate ``y * sigmoid(W2 relu(W1 mean_t y))`` over the band axis of ``[B, C, N, H]``."""

    def __init__(self, bands: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(bands, bands, bias=False), nn.ReLU(), nn.Linear(bands, bands, bias=False)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.sigmoid(self.fc(y.mean(dim=-1))).unsqueeze(-1)


class BandHeads(nn.Module):
    """One ``Linear(L, hidden) -> LeakyReLU -> Linear(hidden, H)`` head per band."""

    def __init__(self, bands: int, length: int, hidden: int, pred_len: int) -> None:
        super().__init__()
        self.first = nn.ModuleList(nn.Linear(length, hidden) for _ in range(bands))
        self.second = nn.ModuleList(nn.Linear(hidden, pred_len) for _ in range(bands))
        self.activation = nn.LeakyReLU()

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        outputs = [
            second(self.activation(first(bands[:, :, index])))
            for index, (first, second) in enumerate(zip(self.first, self.second))
        ]
        return torch.stack(outputs, dim=2)


def _check_options(num_components: int, tree_depth: int, overlap: int) -> None:
    if num_components < 2:
        raise ValueError("num_components must be at least 2")
    if tree_depth < 1:
        raise ValueError("tree_depth must be at least 1")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")


def _segment_lengths(seq_len: int, tree_depth: int, overlap: int) -> list[int]:
    """Length of the halves produced at each internal level (official ``calculate_level_sizes``)."""
    lengths = []
    length = seq_len
    for _ in range(tree_depth):
        if 2 * split_length(length, overlap) < length:
            raise ValueError("without overlap every split segment length must be even")
        length = split_length(length, overlap)
        if length < 3:
            raise ValueError("seq_len is too short for this tree_depth (segments need >= 3 steps)")
        lengths.append(length)
    return lengths


class Model(nn.Module):
    """PRISM forecaster (channel-independent, weights shared across variates)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_components: int = 5,
        tree_depth: int = 2,
        overlap: int = 8,
        use_last_layer_only: bool = False,
    ) -> None:
        super().__init__()
        _check_options(num_components, tree_depth, overlap)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_components = num_components
        self.tree_depth = tree_depth
        self.overlap = overlap
        self.use_last_layer_only = use_last_layer_only

        lengths = _segment_lengths(seq_len, tree_depth, overlap)
        self.segment_lengths = tuple(lengths)
        # Levels whose bands reach the forecast: only the root level when
        # ``use_last_layer_only`` (the official flag name), otherwise all internal levels.
        self.forecast_levels = 1 if use_last_layer_only else tree_depth

        # One router per internal node; level l holds 2**l nodes.
        self.routers = nn.ModuleList(
            nn.ModuleList(BandRouter() for _ in range(2**level)) for level in range(self.forecast_levels)
        )
        bands_per_node = 2 * num_components
        self.heads = nn.ModuleList(
            BandHeads(2**level * bands_per_node, lengths[level], FORECAST_HIDDEN, pred_len)
            for level in range(self.forecast_levels)
        )
        self.total_bands = bands_per_node * (2**self.forecast_levels - 1)
        self.gate = SqueezeExcitationGate(self.total_bands)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-series z-score over time with the sample std floored at ``NORM_EPS``."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(NORM_EPS)
        return (x - mean) / std, mean, std

    def routed_levels(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Weighted bands of every forecasting level, ``[B, C, 2**l * 2K, n_l]`` each.

        Each node splits its input into overlapping halves, decomposes both, weights
        the bands with its router, and hands each half's band-weighted sum to its
        child; the bands of one level are concatenated left to right.
        """
        levels = []
        inputs = [x]
        for level in range(self.forecast_levels):
            weighted, children = [], []
            for node, segment in enumerate(inputs):
                left, right = split_halves(segment, self.overlap)
                left_bands = decompose(left, self.num_components)
                right_bands = decompose(right, self.num_components)
                left_weight, right_weight = self.routers[level][node](left_bands, right_bands)
                left_bands = left_bands * left_weight.unsqueeze(-1)
                right_bands = right_bands * right_weight.unsqueeze(-1)
                weighted.extend((left_bands, right_bands))
                children.extend((left_bands.sum(dim=2), right_bands.sum(dim=2)))
            levels.append(torch.cat(weighted, dim=2))
            inputs = children
        return levels

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must be [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        series, mean, std = self.normalize(x_enc.transpose(1, 2))
        forecasts = [
            heads(bands) for heads, bands in zip(self.heads, self.routed_levels(series))
        ]
        output = self.gate(torch.cat(forecasts, dim=2)).sum(dim=2)
        return (output * std + mean).transpose(1, 2)

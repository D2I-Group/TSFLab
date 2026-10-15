"""Fixed-filter discrete wavelet analysis and synthesis for BCL tensors.

Two paper-neutral transforms, both channel-independent (grouped convolution)
over ``(batch, channels, length)`` tensors:

- :class:`DecimatedWaveletTransform` -- critically sampled (stride-2),
  multi-level orthogonal wavelet decomposition, with an exact linear inverse
  for two-tap orthonormal filters (``haar``/``db1``).
- :class:`UndecimatedWaveletTransform` -- a-trous (dilated, stride-1)
  stationary wavelet decomposition that keeps the sequence length fixed at
  every level, appropriate when downstream code needs aligned per-timestep
  subbands.

Filter taps are the closed-form (``haar``/``db2``) or standard tabulated
(``db4``) orthonormal Daubechies coefficients; these are independently
derivable mathematical constants, entered here directly rather than sourced
from any external package's implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_FILTERS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "haar": (
        (2**-0.5, 2**-0.5),
        (2**-0.5, -(2**-0.5)),
    ),
    "db1": (
        (2**-0.5, 2**-0.5),
        (2**-0.5, -(2**-0.5)),
    ),
    "db2": (
        (
            (1 + 3**0.5) / (4 * 2**0.5),
            (3 + 3**0.5) / (4 * 2**0.5),
            (3 - 3**0.5) / (4 * 2**0.5),
            (1 - 3**0.5) / (4 * 2**0.5),
        ),
        (
            (1 - 3**0.5) / (4 * 2**0.5),
            -(3 - 3**0.5) / (4 * 2**0.5),
            (3 + 3**0.5) / (4 * 2**0.5),
            -(1 + 3**0.5) / (4 * 2**0.5),
        ),
    ),
    "db4": (
        (
            -0.010597401785069032,
            0.032883011666982945,
            0.030841381835986965,
            -0.18703481171888114,
            -0.02798376941698385,
            0.6308807679295904,
            0.7148465705525415,
            0.23037781330885523,
        ),
        (
            -0.23037781330885523,
            0.7148465705525415,
            -0.6308807679295904,
            -0.02798376941698385,
            0.18703481171888114,
            0.030841381835986965,
            -0.032883011666982945,
            -0.010597401785069032,
        ),
    ),
}


def available_wavelets() -> tuple[str, ...]:
    """Return the supported wavelet names."""
    return tuple(sorted(_FILTERS))


def _filters(wavelet: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        low, high = _FILTERS[wavelet]
    except KeyError as exc:
        raise ValueError(f"unsupported wavelet {wavelet!r}; choose one of {available_wavelets()}") from exc
    return torch.tensor(low, dtype=torch.float32), torch.tensor(high, dtype=torch.float32)


class DecimatedWaveletTransform(nn.Module):
    """Critically sampled multi-level orthogonal wavelet analysis/synthesis.

    ``decompose(x)`` maps ``x: (batch, channels, length)`` to
    ``[approx_L, detail_L, ..., detail_1]`` (coarsest first, the common
    ``pywt.wavedec`` ordering). ``reconstruct`` inverts that mapping exactly
    for two-tap orthonormal filters (``haar``/``db1``); longer filters may be
    decomposed but do not support reconstruction here.

    Odd lengths are replicate-padded by one sample per level, and the inverse
    must trim that sample again. ``decompose`` records the per-level flags
    (outermost level first) in ``last_trims``; ``reconstruct`` uses them unless
    an explicit ``trims`` is passed. Without a prior ``decompose`` and without
    ``trims`` all flags are zero, which is exact only for coefficients whose
    original lengths were even at every level. The module is therefore stateful:
    pass ``trims`` explicitly when reconstructing coefficients that did not come
    from this module's most recent ``decompose`` call.
    """

    def __init__(self, wavelet: str = "haar", level: int = 1) -> None:
        super().__init__()
        if level < 1:
            raise ValueError("level must be positive")
        low, high = _filters(wavelet)
        self.register_buffer("low", low.reshape(1, 1, -1))
        self.register_buffer("high", high.reshape(1, 1, -1))
        self.filter_len = low.numel()
        self.level = level
        self.wavelet = wavelet
        self.last_trims: list[int] = []

    def _analysis_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch, channels, length = x.shape
        pad_odd = length % 2
        if pad_odd:
            x = F.pad(x, (0, 1), mode="replicate")
        pad = max(0, self.filter_len - 2)
        if pad:
            x = F.pad(x, (pad, pad), mode="circular")
        flat = x.reshape(batch * channels, 1, -1)
        low = F.conv1d(flat, self.low, stride=2).reshape(batch, channels, -1)
        high = F.conv1d(flat, self.high, stride=2).reshape(batch, channels, -1)
        return low, high, pad_odd

    def decompose(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return ``[approx_L, detail_L, ..., detail_1]`` for ``x``."""
        approx = x
        details = []
        trims: list[int] = []
        for _ in range(self.level):
            approx, detail, trimmed = self._analysis_step(approx)
            details.append(detail)
            trims.append(trimmed)
        self.last_trims = trims
        return [approx, *reversed(details)]

    def _synthesis_step(self, low: torch.Tensor, high: torch.Tensor, trimmed: int) -> torch.Tensor:
        batch, channels, _ = low.shape
        low_flat = low.reshape(batch * channels, 1, -1)
        high_flat = high.reshape(batch * channels, 1, -1)
        merged = F.conv_transpose1d(low_flat, self.low, stride=2) + F.conv_transpose1d(high_flat, self.high, stride=2)
        merged = merged.reshape(batch, channels, -1)
        if trimmed:
            merged = merged[..., :-1]
        return merged

    def reconstruct(self, coeffs: list[torch.Tensor], trims: list[int] | None = None) -> torch.Tensor:
        """Invert :meth:`decompose`; ``coeffs`` must match its output layout.

        ``trims`` is the per-level odd-length flag list (outermost level
        first, length ``level``); default is ``last_trims`` from the most
        recent ``decompose`` (all zeros if there was none).
        """
        if self.filter_len != 2:
            raise NotImplementedError(
                "exact reconstruction is only implemented for two-tap orthonormal filters (haar/db1)"
            )
        if len(coeffs) != self.level + 1:
            raise ValueError(f"expected {self.level + 1} coefficient tensors, got {len(coeffs)}")
        if trims is None:
            trims = self.last_trims or [0] * self.level
        elif len(trims) != self.level:
            raise ValueError(f"expected {self.level} trim flags, got {len(trims)}")
        approx = coeffs[0]
        # ``coeffs[1:]`` is innermost-detail-first (as produced by ``decompose``);
        # ``trims`` was recorded outermost-first, so pair them in reverse.
        for detail, trimmed in zip(coeffs[1:], reversed(trims)):
            approx = self._synthesis_step(approx, detail, trimmed)
        return approx


class UndecimatedWaveletTransform(nn.Module):
    """A-trous stationary wavelet decomposition (no downsampling).

    ``forward(x)`` maps ``x: (batch, channels, length)`` to
    ``[approx_L, detail_L, ..., detail_1]`` where every tensor has the same
    trailing length as ``x``. Each level doubles the filter dilation, so the
    circular receptive field grows exponentially while the sequence length
    stays fixed -- the standard algorithme a trous construction.
    """

    def __init__(self, wavelet: str = "db4", level: int = 3) -> None:
        super().__init__()
        if level < 1:
            raise ValueError("level must be positive")
        low, high = _filters(wavelet)
        self.register_buffer("low", low.reshape(1, 1, -1))
        self.register_buffer("high", high.reshape(1, 1, -1))
        self.filter_len = low.numel()
        self.level = level
        self.wavelet = wavelet

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch, channels, length = x.shape
        flat = x.reshape(batch * channels, 1, length)
        approx = flat
        details = []
        for level_index in range(self.level):
            dilation = 2**level_index
            pad = (self.filter_len - 1) * dilation
            padded = F.pad(approx, (pad, 0), mode="circular")
            low = F.conv1d(padded, self.low, dilation=dilation)
            high = F.conv1d(padded, self.high, dilation=dilation)
            details.append(high.reshape(batch, channels, length))
            approx = low
        approx = approx.reshape(batch, channels, length)
        return [approx, *reversed(details)]

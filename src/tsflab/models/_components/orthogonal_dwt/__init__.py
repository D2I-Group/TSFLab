"""Orthogonal discrete wavelet transform with explicit boundary modes.

A paper-neutral filter-bank DWT in the PyWavelets / ``pytorch_wavelets``
convention: one analysis level computes

    a[n] = sum_j dec_lo[j] x[2n + 1 - j],    d[n] = sum_j dec_hi[j] x[2n + 1 - j],

with ``x`` extended outside ``[0, N)`` by the selected boundary mode, giving
``floor((N + L - 1) / 2)`` coefficients per band for an ``L``-tap filter. One
synthesis level is the transposed operation with the reconstruction filters
(the reversed decomposition filters), keeping ``2n - L + 2`` samples. Two modes
are supported:

- ``"zero"``: ``x`` is zero outside ``[0, N)``.
- ``"symmetric"``: half-sample symmetric extension (edge samples repeated),
  which needs ``N >= L - 1``.

Filter taps are the standard tabulated orthonormal low-pass filters; the
high-pass and reconstruction filters follow from the quadrature-mirror relations
in :func:`filter_bank`. These are independently derivable mathematical
constants, not sourced from any package's implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: Orthonormal decomposition low-pass filters ``dec_lo`` (standard tabulated values).
DEC_LO: dict[str, tuple[float, ...]] = {
    "haar": (0.7071067811865476, 0.7071067811865476),
    "db4": (
        -0.010597401785069032,
        0.0328830116668852,
        0.030841381835560764,
        -0.18703481171909309,
        -0.027983769416859854,
        0.6308807679298589,
        0.7148465705529157,
        0.2303778133088965,
    ),
    "sym3": (
        0.035226291882100656,
        -0.08544127388224149,
        -0.13501102001039084,
        0.4598775021193313,
        0.8068915093133388,
        0.3326705529509569,
    ),
    "coif3": (
        -3.459977319727278e-05,
        -7.0983302506379e-05,
        0.0004662169598204029,
        0.0011175187708306303,
        -0.0025745176881367972,
        -0.009007976136730624,
        0.015880544863669452,
        0.03455502757329774,
        -0.08230192710629983,
        -0.07179982161915484,
        0.42848347637737,
        0.7937772226260872,
        0.40517690240911824,
        -0.06112339000297255,
        -0.06577191128146936,
        0.023452696142077168,
        0.007782596425672746,
        -0.003793512864380802,
    ),
}

#: Supported boundary extension modes.
BOUNDARY_MODES = ("zero", "symmetric")


def filter_bank(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dec_lo, dec_hi, rec_lo, rec_hi)`` in float64 with ``dec_hi[k] = (-1)^(k+1) dec_lo[L-1-k]``
    and the reconstruction filters the reversed decomposition filters."""
    if name not in DEC_LO:
        raise ValueError(f"unsupported wavelet {name!r}; choose one of {sorted(DEC_LO)}")
    dec_lo = torch.tensor(DEC_LO[name], dtype=torch.float64)
    length = dec_lo.numel()
    signs = torch.tensor([(-1.0) ** (k + 1) for k in range(length)], dtype=torch.float64)
    dec_hi = signs * dec_lo.flip(0)
    return dec_lo, dec_hi, dec_lo.flip(0), dec_hi.flip(0)


def coefficient_length(length: int, filter_length: int) -> int:
    """Coefficients per band of one analysis level: ``floor((N + L - 1) / 2)``."""
    return (length + filter_length - 1) // 2


def coefficient_lengths(length: int, filter_length: int, levels: int) -> list[int]:
    """Lengths of ``[d_1, ..., d_J, a_J]`` for a ``levels``-level decomposition."""
    lengths = []
    for _ in range(levels):
        length = coefficient_length(length, filter_length)
        lengths.append(length)
    return lengths + [lengths[-1]]


class OrthogonalDWT(nn.Module):
    """Multi-level orthogonal DWT and its inverse along the last axis.

    ``analyze``/``synthesize`` are one level on ``[..., N]``; ``decompose`` and
    ``reconstruct`` iterate them on the approximation, finest detail first:
    ``[d_1, ..., d_J, a_J]``. ``reconstruct`` drops the last sample of an
    approximation that is one longer than the next detail (odd lengths), so its
    output has length ``2 len(d_1) - L + 2`` (the input length, or one more for an
    odd input; crop it to the original length).

    The filters live in two buffers, ``analysis`` (reversed decomposition
    filters, because ``conv1d`` correlates) and ``synthesis`` (reconstruction
    filters), each ``[2, 1, L]`` (low-pass row first). ``persistent`` decides
    whether they appear in ``state_dict()``.
    """

    def __init__(self, wavelet: str, levels: int = 1, mode: str = "zero", persistent: bool = True) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        if mode not in BOUNDARY_MODES:
            raise ValueError(f"mode must be one of {BOUNDARY_MODES}, got {mode!r}")
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank(wavelet)
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        self.filter_length = dec_lo.numel()
        self.register_buffer(
            "analysis", torch.stack((dec_lo.flip(0), dec_hi.flip(0))).unsqueeze(1).float(), persistent=persistent
        )
        self.register_buffer("synthesis", torch.stack((rec_lo, rec_hi)).unsqueeze(1).float(), persistent=persistent)

    def coefficient_length(self, length: int) -> int:
        """Coefficients per band of one analysis level of a length-``length`` signal."""
        return coefficient_length(length, self.filter_length)

    def _extend(self, flat: torch.Tensor) -> torch.Tensor:
        """Boundary extension of ``[M, 1, N]`` aligned for a stride-2 correlation."""
        length = flat.shape[-1]
        size = self.filter_length
        if self.mode == "zero":
            return F.pad(flat, (size - 2, 2 * coefficient_length(length, size) - length))
        if length < size - 1:
            raise ValueError(
                f"symmetric extension of a {size}-tap filter needs at least {size - 1} samples, got {length}"
            )
        edge = size - 1
        extended = torch.cat((flat[..., :edge].flip(-1), flat, flat[..., -edge:].flip(-1)), dim=-1)
        return extended[..., 1:]

    def analyze(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One level: ``[..., N] -> (a, d)``, each ``[..., floor((N + L - 1) / 2)]``."""
        lead, length = x.shape[:-1], x.shape[-1]
        flat = x.reshape(-1, 1, length)
        coeffs = F.conv1d(self._extend(flat), self.analysis.to(x.dtype), stride=2)
        low, high = coeffs[:, 0], coeffs[:, 1]
        return low.reshape(*lead, -1), high.reshape(*lead, -1)

    def synthesize(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """One inverse level: ``(a, d)`` each ``[..., n]`` -> ``[..., 2n - L + 2]``."""
        lead, size = low.shape[:-1], low.shape[-1]
        weight = self.synthesis.to(low.dtype)
        pad = self.filter_length - 2
        low_flat = low.reshape(-1, 1, size)
        high_flat = high.reshape(-1, 1, high.shape[-1])
        out = F.conv_transpose1d(low_flat, weight[0:1], stride=2, padding=pad) + F.conv_transpose1d(
            high_flat, weight[1:2], stride=2, padding=pad
        )
        return out.reshape(*lead, -1)

    def decompose(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``[..., N] -> [d_1, ..., d_J, a_J]`` (finest detail first)."""
        coefficients = []
        approx = x
        for _ in range(self.levels):
            approx, detail = self.analyze(approx)
            coefficients.append(detail)
        coefficients.append(approx)
        return coefficients

    def reconstruct(self, coefficients: list[torch.Tensor]) -> torch.Tensor:
        """Inverse of :meth:`decompose` for ``[d_1, ..., d_J, a_J]``."""
        if len(coefficients) != self.levels + 1:
            raise ValueError(f"expected {self.levels + 1} coefficient tensors, got {len(coefficients)}")
        approx = coefficients[-1]
        for detail in reversed(coefficients[:-1]):
            if approx.shape[-1] > detail.shape[-1]:
                approx = approx[..., :-1]
            approx = self.synthesize(approx, detail)
        return approx


__all__ = [
    "BOUNDARY_MODES",
    "DEC_LO",
    "OrthogonalDWT",
    "coefficient_length",
    "coefficient_lengths",
    "filter_bank",
]

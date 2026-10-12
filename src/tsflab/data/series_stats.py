"""Pure-numpy time-series statistics shared by dataset inspection and analysis.

The TFB characteristics table and the profile of ``tsf data analyze`` both use
this one implementation. Everything here is deterministic and numpy-only
(statsmodels is optional for the ADF p-value).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # optional; never a hard dependency
    from statsmodels.tsa.stattools import adfuller as _adfuller

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - import guard
    _adfuller = None
    _HAS_STATSMODELS = False


def dominant_period(x: np.ndarray, max_period: Optional[int] = None) -> int:
    """Estimate the dominant seasonal period via the FFT power spectrum.

    Returns an integer period >= 2. Falls back to 2 when no clear peak exists.
    """
    n = x.size
    if n < 4:
        return 2
    x = x - x.mean()
    if not np.any(x):
        return 2
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Ignore the zero-frequency (DC) component.
    spectrum[0] = 0.0
    cap = max_period if max_period is not None else n // 2
    # Drop ultra-low frequencies whose period exceeds the cap (treated as trend).
    for i in range(1, freqs.size):
        if freqs[i] > 0 and (1.0 / freqs[i]) > cap:
            spectrum[i] = 0.0
    if not np.any(spectrum):
        return 2
    peak = int(np.argmax(spectrum))
    if freqs[peak] <= 0:
        return 2
    period = int(round(1.0 / freqs[peak]))
    return max(2, period)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding (returns same length)."""
    if window < 2:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def seasonal_component(detrended: np.ndarray, period: int) -> np.ndarray:
    """Period-averaged seasonal estimate, broadcast back over the series."""
    n = detrended.size
    if period < 2 or period >= n:
        return np.zeros_like(detrended)
    means = np.zeros(period, dtype=np.float64)
    counts = np.zeros(period, dtype=np.float64)
    idx = np.arange(n) % period
    np.add.at(means, idx, detrended)
    np.add.at(counts, idx, 1.0)
    counts = np.where(counts == 0, 1.0, counts)
    means = means / counts
    means = means - means.mean()  # center so the seasonal sums to ~0
    return means[idx]


def strength(var_resid: float, var_combined: float) -> float:
    """STL-style strength: ``max(0, 1 - Var(resid) / Var(resid + comp))``."""
    if var_combined <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - var_resid / var_combined)))


def trend_and_seasonal_strength(x: np.ndarray, period: int) -> tuple[float, float]:
    n = x.size
    if n < 4:
        return 0.0, 0.0
    # Trend via moving average over (roughly) one seasonal cycle.
    trend_window = max(3, period)
    trend = moving_average(x, trend_window)
    detrended = x - trend
    seasonal = seasonal_component(detrended, period)
    resid = detrended - seasonal

    # Trend strength: how much variance the moving-average trend removes from x
    # (resid+seasonal == detrended vs the original series).
    trend_strength = strength(np.var(detrended), np.var(x))
    # Seasonality strength: how much variance the seasonal component removes from
    # the detrended series (resid vs resid+seasonal).
    seasonal_strength = strength(np.var(resid), np.var(resid + seasonal))
    return trend_strength, seasonal_strength


def stationarity(x: np.ndarray) -> tuple[float, str]:
    """Stationarity score in ``[0, 1]`` plus a label describing the method.

    Uses the ADF test p-value (statsmodels) when available, otherwise a
    lightweight rolling mean/variance stability ratio. Higher is more stationary.
    """
    n = x.size
    if n < 8:
        return 0.0, "n/a"
    if _HAS_STATSMODELS:
        try:
            pvalue = float(_adfuller(x, autolag="AIC")[1])
            # Map p-value to a 0..1 "stationarity" score (1 - p): small p => stationary.
            return float(max(0.0, min(1.0, 1.0 - pvalue))), "adf(1-pvalue)"
        except Exception:
            pass
    # Lightweight fallback: compare rolling means/vars across windows. A series
    # whose windowed moments barely move scores near 1.0.
    n_windows = min(10, max(2, n // 20))
    splits = np.array_split(x, n_windows)
    means = np.array([s.mean() for s in splits])
    variances = np.array([s.var() for s in splits])
    global_std = x.std()
    if global_std <= 1e-12:
        return 1.0, "rolling(n/a-statsmodels)"
    mean_drift = means.std() / (abs(x.mean()) + global_std + 1e-12)
    var_global = x.var()
    var_drift = variances.std() / (var_global + 1e-12)
    score = 1.0 / (1.0 + mean_drift + var_drift)
    return float(max(0.0, min(1.0, score))), "rolling(n/a-statsmodels)"


def shifting(x: np.ndarray) -> float:
    """Absolute mean shift between halves, normalised by the series std."""
    n = x.size
    if n < 4:
        return 0.0
    half = n // 2
    first, second = x[:half], x[half:]
    std = x.std()
    if std <= 1e-12:
        return 0.0
    return float(abs(second.mean() - first.mean()) / std)


def transition(x: np.ndarray) -> float:
    """Lag-1 autocorrelation."""
    n = x.size
    if n < 2:
        return 0.0
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc[:-1], xc[1:]) / denom)


def channel_correlation(series: np.ndarray) -> float:
    """Mean absolute off-diagonal pairwise channel correlation."""
    if series.shape[1] < 2:
        return float("nan")
    # Guard against zero-variance channels (corrcoef yields NaN rows).
    stds = series.std(axis=0)
    keep = stds > 1e-12
    if keep.sum() < 2:
        return float("nan")
    corr = np.corrcoef(series[:, keep], rowvar=False)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = np.abs(corr[mask])
    off = off[np.isfinite(off)]
    if off.size == 0:
        return float("nan")
    return float(off.mean())

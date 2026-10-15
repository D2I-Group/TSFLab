"""Split-aware statistical profile of a multivariate series (numpy/scipy only).

``build_profile`` takes the chronological train/val/test arrays of one dataset and
returns a JSON-serializable profile an Agent reads before choosing models. The
profile is split by what may be used for decisions:

* ``train`` metrics and ``shift.within_train`` / ``shift.train_to_val`` are decision
  inputs (lookback candidates and catalog rules use only these).
* ``shift.train_to_test`` is a diagnostic: it describes the test split and must
  never drive model, hyperparameter, or rule selection.

Rules that map the profile to catalog facts live in ``profile_rules.toml``.
"""

from __future__ import annotations

import math
import tomllib
import warnings
from collections.abc import Mapping
from importlib import resources
from typing import Any

import numpy as np

from tsflab.data import series_stats as ss

SCHEMA = "tsflab.dataset-profile/1"
DECISION_PREFIXES = ("dataset.", "split.", "train.", "shift.within_train.", "shift.train_to_val.")
STANDARD_LOOKBACKS = (96, 192, 336, 512, 720)
_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _f(value: Any, digits: int = 4) -> float | None:
    value = float(value)
    return None if not math.isfinite(value) else round(value, digits)


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _sample_columns(n_channels: int, limit: int) -> np.ndarray:
    if n_channels <= limit:
        return np.arange(n_channels)
    return np.unique(np.linspace(0, n_channels - 1, limit).round().astype(int))


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Replace NaN/inf by the channel's finite mean (0 for all-missing channels)."""
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    clean = np.where(bad, np.nan, x)
    with np.errstate(all="ignore"):
        means = np.nanmean(clean, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    return np.where(bad, means[None, :], x)


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = x.std(axis=0)
    keep = std > 1e-12
    z = (x[:, keep] - x[:, keep].mean(axis=0)) / std[keep]
    return z, keep


def _detrend_linear(z: np.ndarray) -> np.ndarray:
    n = z.shape[0]
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    slope = (t[:, None] * (z - z.mean(axis=0))).sum(axis=0) / (t @ t)
    return z - z.mean(axis=0) - t[:, None] * slope[None, :]


def _power(z: np.ndarray) -> np.ndarray:
    """Hann-windowed periodogram per channel, shape ``(bins, channels)``."""
    n = z.shape[0]
    win = np.hanning(n)[:, None]
    return np.abs(np.fft.rfft(_detrend_linear(z) * win, axis=0)) ** 2


def _acf(z: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean autocorrelation across channels for lags ``0..max_lag`` via FFT."""
    n = z.shape[0]
    zc = z - z.mean(axis=0)
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(zc, n=size, axis=0)
    ac = np.fft.irfft(spec * np.conj(spec), n=size, axis=0)[: max_lag + 1]
    ac = ac / np.maximum(ac[0:1], 1e-12)
    return ac.mean(axis=1)


# --------------------------------------------------------------------------- #
# Frequency
# --------------------------------------------------------------------------- #
def infer_frequency(timestamps: Any) -> dict[str, Any]:
    """Describe the sampling interval from a datetime-like sequence (or ``None``)."""
    if timestamps is None:
        return {"label": None, "step_seconds": None, "steps_per_day": None}
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce")).dropna()
    if len(idx) < 3:
        return {"label": None, "step_seconds": None, "steps_per_day": None}
    step = float(np.median(np.diff(idx.values).astype("timedelta64[s]").astype(np.int64)))
    if step <= 0:
        return {"label": None, "step_seconds": None, "steps_per_day": None}
    units = ((604800, "W"), (86400, "D"), (3600, "h"), (60, "min"), (1, "s"))
    label = f"{int(step)}s"
    for size, unit in units:
        if step >= size and step % size == 0:
            label = f"{int(step // size)}{unit}"
            break
    return {"label": label, "step_seconds": step, "steps_per_day": _f(86400.0 / step)}


def _period_tag(period: float, steps_per_day: float | None) -> str | None:
    if not steps_per_day:
        return None
    for mult, tag in ((1 / 2, "half-day"), (1, "daily"), (7, "weekly"), (365.25, "yearly")):
        target = steps_per_day * mult
        if target >= 2 and abs(period - target) / target <= 0.06:
            return tag
    return None


# --------------------------------------------------------------------------- #
# Train-split sections
# --------------------------------------------------------------------------- #
def _quality(raw: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(raw)
    clean = np.where(finite, raw, np.nan)
    nan_frac_ch = 1.0 - finite.mean(axis=0)
    valid = clean[finite]
    zero = clean == 0
    repeat = np.zeros_like(finite)
    repeat[1:] = (clean[1:] == clean[:-1]) & (clean[1:] != 0) & finite[1:]
    zero_ch = zero.sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
    return {
        "missing_fraction": _f(1.0 - finite.mean()),
        "channels_with_missing": int((nan_frac_ch > 0).sum()),
        "zero_fraction": _f(zero.sum() / max(valid.size, 1)),
        "channels_mostly_zero": int((zero_ch > 0.5).sum()),
        "repeat_fraction": _f(repeat.sum() / max(valid.size, 1)),
        "negative_fraction": _f((valid < 0).mean() if valid.size else 0.0),
        "constant_channels": int((np.nanstd(clean, axis=0) <= 1e-12).sum()),
    }


def _scale(x: np.ndarray, cols: np.ndarray, block: int) -> dict[str, Any]:
    std = x.std(axis=0)
    mean_abs = np.abs(x).mean(axis=0)
    p10, p90 = np.percentile(std, [10, 90])
    out: dict[str, Any] = {
        "std_ratio_p90_p10": _f(p90 / max(p10, 1e-12)),
        "cv_median": _f(_median(std / np.maximum(mean_abs, 1e-12))),
    }
    xs = x[:, cols]
    nb = max(xs.shape[0] // max(block, 4), 2)
    blocks = np.array_split(xs, nb, axis=0)
    bmean = np.stack([b.mean(axis=0) for b in blocks])
    bstd = np.stack([b.std(axis=0) for b in blocks])
    cv_std = bstd.std(axis=0) / np.maximum(bstd.mean(axis=0), 1e-12)
    dep = []
    for j in range(xs.shape[1]):
        if bmean[:, j].std() > 1e-12 and bstd[:, j].std() > 1e-12:
            dep.append(np.corrcoef(bmean[:, j], bstd[:, j])[0, 1])
    centered = xs - xs.mean(axis=0)
    s = np.maximum(xs.std(axis=0), 1e-12)
    skew = (centered**3).mean(axis=0) / s**3
    kurt = (centered**4).mean(axis=0) / s**4 - 3.0
    out.update(
        heteroscedasticity=_f(_median(cv_std)),
        level_std_correlation=_f(_median(np.array(dep))) if dep else None,
        skewness_median=_f(_median(skew)),
        excess_kurtosis_median=_f(_median(kurt)),
        positive_skew_nonnegative=bool(x.min() >= 0 and _median(skew) > 1.0),
    )
    return out


def _periods(z: np.ndarray, steps_per_day: float | None, top_k: int = 5) -> dict[str, Any]:
    n = z.shape[0]
    mean_p, ent, forecastability = _spectrum(z)
    max_lag = min(n // 3, 4096)
    acf = _acf(z, max_lag)
    peaks = _spectral_peaks(mean_p, n, acf, max_lag, steps_per_day, top_k)
    below = np.where(acf < 0.1)[0]
    return {
        "dominant": peaks,
        "n_strong": int(sum(1 for p in peaks if p["harmonic_of"] is None and (p["energy_share"] or 0) >= 0.03)),
        "acf_lag1": _f(acf[1]) if acf.size > 1 else None,
        "acf_decay_lag": int(below[0]) if below.size else None,
        "acf_max_lag": int(max_lag),
        "forecastability": {
            "spectral_entropy_median": _f(_median(ent)),
            "score": _f(_median(forecastability)),
        },
    }


def _spectrum(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-mean normalized power, per-channel spectral entropy, and forecastability."""
    power = _power(z)
    power = power / np.maximum(power.sum(axis=0, keepdims=True), 1e-12)
    mean_p = power.mean(axis=1)
    # Spectral entropy per channel; forecastability = 1 - normalized entropy.
    body = power[1:]
    body = body / np.maximum(body.sum(axis=0, keepdims=True), 1e-12)
    ent = -(body * np.log(np.maximum(body, 1e-18))).sum(axis=0) / math.log(body.shape[0])
    forecastability = 1.0 - ent
    return mean_p, ent, forecastability


def _spectral_peaks(
    mean_p: np.ndarray, n: int, acf: np.ndarray, max_lag: int, steps_per_day: float | None, top_k: int
) -> list[dict[str, Any]]:
    lo_bin = 3  # need >= 3 cycles in the window
    hi_bin = mean_p.size - 1
    floor = float(np.median(mean_p[lo_bin : hi_bin + 1])) or 1e-12
    cand = [k for k in range(lo_bin, hi_bin) if mean_p[k] >= mean_p[k - 1] and mean_p[k] >= mean_p[k + 1]]
    cand.sort(key=lambda k: -mean_p[k])
    peaks: list[dict[str, Any]] = []
    for k in cand:
        period = n / k
        if period < 2 or _near_known(period, peaks):
            continue
        lag = int(round(period))
        lag_ok = 2 <= lag <= max_lag
        peaks.append(
            {
                "period": _f(period, 2),
                "peak_ratio": _f(mean_p[k] / floor, 2),
                "energy_share": _f(mean_p[k] / max(mean_p[lo_bin:].sum(), 1e-12)),
                "acf_at_period": _f(acf[max(lag - 1, 0) : lag + 2].max()) if lag_ok else None,
                "tag": _period_tag(period, steps_per_day),
                "harmonic_of": _harmonic_of(period, peaks),
            }
        )
        if len(peaks) == top_k:
            break
    return peaks


def _near_known(period: float, peaks: list[dict[str, Any]]) -> bool:
    return any(abs(period - p["period"]) / p["period"] < 0.1 for p in peaks)


def _harmonic_of(period: float, peaks: list[dict[str, Any]]) -> float | None:
    return next(
        (q["period"] for q in peaks for m in range(2, 9) if abs(period - q["period"] / m) / period < 0.06),
        None,
    )


def _trend_season(x: np.ndarray, cols: np.ndarray, period: int | None) -> tuple[dict, dict, dict]:
    n = x.shape[0]
    if not period or period < 2 or period >= n // 2:
        period = max(2, min(24, n // 8))
        seasonal_valid = False
    else:
        seasonal_valid = True
    tr, se = [], []
    for j in cols:
        t, s = ss.trend_and_seasonal_strength(x[:, j], int(period))
        tr.append(t)
        se.append(s)
    stat, drift_mean, drift_var = _drift(x, cols)
    trend = {"strength": _f(_median(np.array(tr))), "period_used": int(period)}
    seasonal = {
        "strength": _f(_median(np.array(se))) if seasonal_valid else 0.0,
        "period_used": int(period) if seasonal_valid else None,
        "share_channels_ge_0_5": _f(np.mean(np.array(se) >= 0.5)) if seasonal_valid else 0.0,
    }
    stationarity = {
        "score": _f(_median(np.array(stat))),
        "method": "rolling-moments" if not ss._HAS_STATSMODELS else "adf(1-pvalue)",
        "mean_drift": _f(_median(np.array(drift_mean))),
        "variance_drift": _f(_median(np.array(drift_var))),
    }
    return trend, seasonal, stationarity


def _drift(x: np.ndarray, cols: np.ndarray) -> tuple[list, list, list]:
    """Per-channel stationarity score and block mean/variance drift."""
    stat, drift_mean, drift_var = [], [], []
    for j in cols:
        sc, _ = ss.stationarity(x[:, j])
        stat.append(sc)
        blocks = np.array_split(x[:, j], 10)
        m = np.array([b.mean() for b in blocks])
        v = np.array([b.var() for b in blocks])
        drift_mean.append(m.std() / max(x[:, j].std(), 1e-12))
        drift_var.append(v.std() / max(v.mean(), 1e-12))
    return stat, drift_mean, drift_var


def _cross_channel(x: np.ndarray, limit: int = 256) -> dict[str, Any]:
    c = x.shape[1]
    if c < 2:
        return {"applicable": False}
    xs = x[:, _sample_columns(c, limit)]

    def corr_stats(m: np.ndarray) -> tuple[float, float, float, float]:
        z, keep = _standardize(m)
        if keep.sum() < 2:
            return (float("nan"),) * 4
        cm = np.corrcoef(z, rowvar=False)
        off = np.abs(cm[~np.eye(cm.shape[0], dtype=bool)])
        off = off[np.isfinite(off)]
        eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(cm)), 0, None)
        p = eig / max(eig.sum(), 1e-12)
        eff_rank = math.exp(-(p[p > 0] * np.log(p[p > 0])).sum())
        return float(off.mean()), float((off > 0.7).mean()), float(eig.max() / eig.sum()), eff_rank / cm.shape[0]

    lv = corr_stats(xs)
    df = corr_stats(np.diff(xs, axis=0))
    return {
        "applicable": True,
        "channels_sampled": int(xs.shape[1]),
        "mean_abs_corr": _f(lv[0]),
        "high_corr_pair_fraction": _f(lv[1]),
        "pc1_share": _f(lv[2]),
        "effective_rank_ratio": _f(lv[3]),
        "mean_abs_corr_diff": _f(df[0]),
        "pc1_share_diff": _f(df[2]),
    }


def _anomalies(x: np.ndarray, cols: np.ndarray, block: int) -> dict[str, Any]:
    xs = x[:, cols]
    med = np.median(xs, axis=0)
    mad = np.median(np.abs(xs - med), axis=0) * 1.4826
    ok = mad > 1e-12
    frac = (np.abs(xs[:, ok] - med[ok]) / mad[ok] > 5).mean(axis=0) if ok.any() else np.array([0.0])
    nb = max(xs.shape[0] // max(block, 4), 4)
    bm = np.stack([b.mean(axis=0) for b in np.array_split(xs, nb, axis=0)])
    jumps = np.abs(np.diff(bm, axis=0))
    jscale = np.maximum(np.median(jumps, axis=0) * 1.4826, 1e-12)
    std = np.maximum(xs.std(axis=0), 1e-12)
    return {
        "outlier_fraction_5mad": _f(_median(frac), 5),
        "level_shift_max": _f(_median(jumps.max(axis=0) / std)),
        "level_shift_count": _f(_median(((jumps / jscale) > 6).sum(axis=0)), 1),
        "block_length": int(block),
    }


# --------------------------------------------------------------------------- #
# Shift
# --------------------------------------------------------------------------- #
def _shift(train: np.ndarray, other: np.ndarray, cols: np.ndarray) -> dict[str, Any]:
    from scipy.stats import ks_2samp

    ts = np.maximum(train.std(axis=0), 1e-12)
    mean_shift = np.abs(other.mean(axis=0) - train.mean(axis=0)) / ts
    ratio = np.abs(np.log2(np.maximum(other.std(axis=0), 1e-12) / ts))
    kcols = cols[:32] if cols.size > 32 else cols
    ks = [ks_2samp(train[:, j], other[:, j]).statistic for j in kcols]
    return {
        "mean_shift_median": _f(_median(mean_shift)),
        "mean_shift_p90": _f(np.percentile(mean_shift, 90)),
        "log2_std_ratio_median": _f(_median(ratio)),
        "ks_statistic_median": _f(_median(np.array(ks))),
    }


def _pooled_top_period(z: np.ndarray) -> float | None:
    if z.shape[0] < 16 or z.shape[1] == 0:
        return None
    power = _power(z)
    mean_p = (power / np.maximum(power.sum(axis=0, keepdims=True), 1e-12)).mean(axis=1)
    k = int(np.argmax(mean_p[3:])) + 3
    return z.shape[0] / k


# --------------------------------------------------------------------------- #
# Lookback candidates
# --------------------------------------------------------------------------- #
def lookback_candidates(
    periods: list[dict[str, Any]], train_len: int, val_len: int, acf_decay: int | None
) -> list[dict[str, Any]]:
    """Candidate input lengths from train-only periodicity and capacity limits."""
    cap = int(max(16, min(1024, train_len // 4, max(val_len // 2, 16))))
    out: dict[int, dict[str, Any]] = {}

    def add(value: int, reason: str, source: str) -> None:
        if 8 <= value <= cap and value not in out:
            out[value] = {"value": int(value), "reason": reason, "source": source}

    for p in periods[:2]:
        base = int(round(p["period"]))
        label = p.get("tag") or f"period {base}"
        for mult in (1, 2, 4, 7):
            if mult == 7 and p.get("tag") != "daily":
                continue
            add(base * mult, f"{mult} x {label} ({base} steps)", "period")
    if acf_decay:
        add(int(acf_decay), "first lag where mean ACF falls below 0.1", "acf")
    for value in STANDARD_LOOKBACKS:
        add(value, "common benchmark lookback", "benchmark")
    return sorted(out.values(), key=lambda c: c["value"])


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_profile(
    splits: Mapping[str, np.ndarray],
    *,
    name: str,
    source: str = "",
    frequency: Mapping[str, Any] | None = None,
    max_channels: int = 128,
    split_ratio: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Profile ``splits['train'|'val'|'test']`` (each ``(T, C)``, raw scale)."""
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in splits.items()}
    for key, arr in list(arrays.items()):
        arrays[key] = arr[:, None] if arr.ndim == 1 else arr
    raw_train = arrays["train"]
    n, c = raw_train.shape
    if n < 32:
        raise ValueError(f"train split has only {n} steps; need at least 32 to profile")
    freq = dict(frequency or {"label": None, "step_seconds": None, "steps_per_day": None})
    spd = freq.get("steps_per_day")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_full = _fill_nan(raw_train)
        active = np.where(x_full.std(axis=0) > 1e-12)[0]
        if active.size == 0:
            raise ValueError("every train channel is constant or missing; nothing to profile")
        x = x_full[:, active]
        return _profile_active(arrays, raw_train, x, active, name, source, freq, spd, max_channels, split_ratio)


def _shift_section(arrays, x, active, cols, top, n):
    """Within-train, train-to-val, and diagnostic train-to-test distribution shift."""
    half = n // 2
    within = float(
        _median(
            np.abs(x[half:, cols].mean(axis=0) - x[:half, cols].mean(axis=0))
            / np.maximum(x[:, cols].std(axis=0), 1e-12)
        )
    )
    shift: dict[str, Any] = {"within_train": {"half_mean_shift_median": _f(within)}}
    val = arrays.get("val")
    test = arrays.get("test")
    if val is not None and len(val) >= 8:
        val_x = _fill_nan(val)[:, active]
        shift["train_to_val"] = _shift(x, val_x, cols)
        zv, _ = _standardize(val_x[:, cols])
        pv = _pooled_top_period(zv)
        shift["train_to_val"]["top_period_ratio"] = _f(pv / top) if pv and top else None
    if test is not None and len(test) >= 8:
        shift["train_to_test"] = {
            **_shift(x, _fill_nan(test)[:, active], cols),
            "use": "diagnostic-only; never select models, hyperparameters, or rules from it",
        }
    return shift


def _profile_active(arrays, raw_train, x, active, name, source, freq, spd, max_channels, split_ratio):
    n, c = raw_train.shape
    cols = _sample_columns(x.shape[1], max_channels)
    z, _ = _standardize(x[:, cols])
    periods = _periods(z, spd)
    top = periods["dominant"][0]["period"] if periods["dominant"] else None
    block = int(max(top or 24, n // 40, 8))
    trend, seasonal, stationarity = _trend_season(x, cols, int(round(top)) if top else None)
    train: dict[str, Any] = {
        "quality": _quality(raw_train),
        "scale": _scale(x, cols, block),
        "periods": periods,
        "seasonal": seasonal,
        "trend": trend,
        "stationarity": stationarity,
        "cross_channel": _cross_channel(x),
        "anomalies": _anomalies(x, cols, block),
    }
    shift = _shift_section(arrays, x, active, cols, top, n)
    lengths = {k: int(len(v)) for k, v in arrays.items()}
    return {
        "schema": SCHEMA,
        "dataset": {
            "name": name,
            "source": source,
            "channels": int(c),
            "frequency": freq,
            "split_ratio": list(split_ratio) if split_ratio else None,
            "channels_active": int(active.size),
            "channels_sampled": int(cols.size),
        },
        "split": lengths,
        "train": train,
        "shift": shift,
        "lookback_candidates": lookback_candidates(
            periods["dominant"], lengths["train"], lengths.get("val", 0), periods["acf_decay_lag"]
        ),
        "leakage_policy": (
            "Recommendations use train statistics and train-to-val shift only. "
            "shift.train_to_test is a diagnostic and must not be used for selection."
        ),
    }


# --------------------------------------------------------------------------- #
# Profile -> catalog rules
# --------------------------------------------------------------------------- #
def load_rules() -> dict[str, Any]:
    """Return the packaged profile-to-catalog table (``profile_rules.toml``)."""
    text = resources.files("tsflab.data").joinpath("profile_rules.toml").read_text("utf-8")
    return tomllib.loads(text)


def lookup(profile: Mapping[str, Any], path: str) -> Any:
    node: Any = profile
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _fire(rule: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any] | None:
    observed = lookup(profile, rule["metric"])
    if observed is None or isinstance(observed, bool):
        return None
    if not _OPS[rule["op"]](float(observed), float(rule["value"])):
        return None
    return {
        "id": rule["id"],
        "metric": rule["metric"],
        "observed": observed,
        "condition": f"{rule['op']} {rule['value']}",
        "finding": rule["finding"],
        "hypothesis": rule["hypothesis"],
        "components": list(rule.get("components", [])),
        "models": list(rule.get("models", [])),
        "slots": {k: list(v) for k, v in rule.get("slots", {}).items()},
    }


def _promoted(name: str, fired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    promoted: dict[str, list[str]] = {}
    for item in fired:
        for option in item["slots"].get(name, []):
            promoted.setdefault(option, []).append(item["id"])
    return [{"option": o, "rules": r} for o, r in promoted.items()]


def evaluate_rules(profile: Mapping[str, Any], table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fire rules against ``profile``; return fired rules and merged slot candidates."""
    table = table or load_rules()
    fired = [item for item in (_fire(rule, profile) for rule in table.get("rule", [])) if item is not None]
    slots: dict[str, dict[str, Any]] = {}
    for name, spec in table.get("slot", {}).items():
        slots[name] = {
            "description": spec["description"],
            "defaults": list(spec.get("defaults", [])),
            "promoted": _promoted(name, fired),
        }
    return {"fired": fired, "slots": slots}


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _dominant_text(dominant: list[Mapping[str, Any]]) -> str:
    return (
        ", ".join(
            f"{p['period']}" + (f" ({p['tag']})" if p["tag"] else "") + f" ratio {p['peak_ratio']}"
            for p in dominant[:3]
        )
        or "none"
    )


def _finding_lines(item: Mapping[str, Any]) -> list[str]:
    lines = [f"- **{item['id']}** ({item['metric']} = {item['observed']}): {item['finding']}"]
    if item["components"]:
        lines.append(f"  - components: {', '.join(item['components'])}")
    if item["models"]:
        lines.append(f"  - models: {', '.join(item['models'])}")
    lines.append(f"  - hypothesis: {item['hypothesis']}")
    return lines


def render_markdown(profile: Mapping[str, Any], recommendations: Mapping[str, Any]) -> str:
    d, tr, sh = profile["dataset"], profile["train"], profile["shift"]
    freq = d["frequency"]
    q, sc, pe = tr["quality"], tr["scale"], tr["periods"]
    cc, an = tr["cross_channel"], tr["anomalies"]
    lines = [
        f"# Dataset profile: {d['name']}",
        "",
        f"- Source: `{d['source']}`; channels {d['channels']} "
        f"(sampled {d['channels_sampled']} for spectral statistics)",
        "- Split lengths: "
        + ", ".join(f"{k} {v}" for k, v in profile["split"].items())
        + f"; frequency {freq.get('label') or 'unknown'}",
        f"- Missing {q['missing_fraction']}, zeros {q['zero_fraction']} "
        f"(channels mostly zero: {q['channels_mostly_zero']}), outliers(5 MAD) {an['outlier_fraction_5mad']}",
        f"- Scale: std p90/p10 {sc['std_ratio_p90_p10']}, heteroscedasticity {sc['heteroscedasticity']}, "
        f"level-std corr {sc['level_std_correlation']}",
        "- Dominant periods: " + _dominant_text(pe["dominant"]),
        f"- Seasonal strength {tr['seasonal']['strength']}, trend strength {tr['trend']['strength']}, "
        f"forecastability {pe['forecastability']['score']}, stationarity score {tr['stationarity']['score']} "
        f"(mean drift {tr['stationarity']['mean_drift']})",
        "- Cross-channel: "
        + (
            f"mean |corr| {cc['mean_abs_corr']}, PC1 share {cc['pc1_share']}, effective-rank ratio {cc['effective_rank_ratio']}"
            if cc.get("applicable")
            else "single channel"
        ),
        f"- Level shifts {an['level_shift_count']} (max {an['level_shift_max']} std)",
        f"- Shift within train {sh['within_train']['half_mean_shift_median']}; "
        f"train to val mean {lookup(profile, 'shift.train_to_val.mean_shift_median')}, "
        f"std ratio(log2) {lookup(profile, 'shift.train_to_val.log2_std_ratio_median')}; "
        f"test shift (diagnostic only) {lookup(profile, 'shift.train_to_test.mean_shift_median')}",
        "- Lookback candidates: " + ", ".join(str(c["value"]) for c in profile["lookback_candidates"]),
        "",
        "## Findings and catalog actions",
        "",
    ]
    for item in recommendations["fired"]:
        lines += _finding_lines(item)
    if not recommendations["fired"]:
        lines.append("- No rule fired; start from the baseline panel.")
    lines += ["", f"_{profile['leakage_policy']}_", ""]
    return "\n".join(lines)

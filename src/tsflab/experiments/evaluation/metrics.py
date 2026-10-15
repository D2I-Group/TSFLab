"""Built-in metric implementations and registration."""

from __future__ import annotations

import math

import numpy as np

from tsflab.catalog.registry import METRIC_REGISTRY


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(true - pred)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((true - pred) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute percentage error."""
    return float(np.mean(np.abs((true - pred) / true)))


def mspe(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared percentage error."""
    return float(np.mean(np.square((true - pred) / true)))


def corr(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-channel Pearson correlation between pred and true.

    Arrays are expected to be shaped ``(B, pred_len, C)``. The correlation is
    computed per channel over the flattened ``(B, pred_len)`` samples, then
    averaged across channels. Channels with zero variance contribute 0.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    c = p.shape[-1]
    p = p.reshape(-1, c)
    t = t.reshape(-1, c)
    p_mean = p.mean(axis=0, keepdims=True)
    t_mean = t.mean(axis=0, keepdims=True)
    p_dev = p - p_mean
    t_dev = t - t_mean
    num = (p_dev * t_dev).sum(axis=0)
    denom = np.sqrt((p_dev**2).sum(axis=0) * (t_dev**2).sum(axis=0))
    per_channel = np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-12)
    return float(np.nan_to_num(per_channel).mean())


def rse(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative squared error: ||pred-true||_2 / ||true-mean(true)||_2."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    num = np.sqrt(np.sum((t - p) ** 2))
    denom = np.sqrt(np.sum((t - t.mean()) ** 2))
    return float(num / (denom + 1e-5))


def wape(pred: np.ndarray, true: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|pred-true| / sum|true|."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    return float(np.sum(np.abs(p - t)) / (np.sum(np.abs(t)) + 1e-5))


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    """Symmetric mean absolute percentage error (in percent)."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    val = 2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8)
    return float(np.mean(np.nan_to_num(val)) * 100.0)


def mase(pred: np.ndarray, true: np.ndarray, season: int = 1) -> float:
    """Mean absolute scaled error.

    MASE = mean|pred-true| / mean|naive seasonal error|.

    The seasonal naive error is computed over the concatenated ground-truth
    sequence (flattened across the batch and horizon dimensions) using a lag of
    ``season`` steps (default 1 = last-value naive).

    Limitation: this benchmark only has the prediction window available at
    evaluation time, not the full historical series. The naive baseline is
    therefore derived from the test targets themselves rather than from the
    in-sample training history, so the absolute MASE values are not directly
    comparable to the textbook in-sample-scaled definition. If the denominator
    cannot be computed (too few samples or zero seasonal error) the function
    returns ``float('nan')``.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    season = max(int(season), 1)
    flat = t.reshape(-1)
    if flat.shape[0] <= season:
        return float("nan")
    naive = np.mean(np.abs(flat[season:] - flat[:-season]))
    if not np.isfinite(naive) or naive <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(p - t)) / naive)


def collect_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Compute the default metric suite.

    Parameters
    ----------
    pred : np.ndarray
        Model predictions.
    true : np.ndarray
        Ground-truth targets.

    Returns
    -------
    dict[str, float]
        Metrics keyed by name.
    """
    return {
        "mae": mae(pred, true),
        "mse": mse(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
        "mspe": mspe(pred, true),
        "corr": corr(pred, true),
        "rse": rse(pred, true),
        "wape": wape(pred, true),
        "smape": smape(pred, true),
        "mase": mase(pred, true),
    }


# ---------------------------------------------------------------------------
# Probabilistic metrics (Phase 1)
# ---------------------------------------------------------------------------


def _pinball(yhat_q: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    """Pinball (quantile) loss elementwise for a single quantile level ``q``.

    ``max(q * (y - yhat_q), (q - 1) * (y - yhat_q))``. ``yhat_q`` and ``y`` must
    broadcast together; ``q`` is a scalar float.
    """
    e = y - yhat_q
    return np.maximum(q * e, (q - 1.0) * e)


def _phi_inv(p: np.ndarray) -> np.ndarray:
    """Standard-normal inverse CDF (quantile function).

    Uses ``scipy.special.ndtri`` when scipy is importable; otherwise falls back
    to a numpy ``erfinv`` implementation via ``sqrt(2) * erfinv(2p - 1)``.
    """
    try:
        from scipy.special import ndtri

        return ndtri(np.asarray(p, dtype=np.float64))
    except Exception:  # pragma: no cover - scipy fallback
        erfinv = np.vectorize(_erfinv_scalar)
        return math.sqrt(2.0) * erfinv(2.0 * np.asarray(p, dtype=np.float64) - 1.0)


def _erfinv_scalar(x: float) -> float:  # pragma: no cover - scipy fallback path
    """Scalar inverse error function (used only when scipy is unavailable)."""
    a = 0.147
    ln = math.log(1.0 - x * x)
    term = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), x)


def _phi_cdf(x: np.ndarray) -> np.ndarray:
    """Standard-normal CDF: ``0.5 * (1 + erf(x / sqrt(2)))``."""
    from scipy.special import erf

    return 0.5 * (1.0 + erf(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))


def _phi_pdf(x: np.ndarray) -> np.ndarray:
    """Standard-normal PDF: ``exp(-x^2 / 2) / sqrt(2 * pi)``."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _quantile_grid(
    pred: np.ndarray,
    levels: list[float],
    output_type: str,
) -> np.ndarray:
    """Return a ``(B, L, C, Q)`` quantile prediction grid.

    For ``"quantile"`` models ``pred`` already is the grid. For ``"distribution"``
    (gaussian) models the quantiles are derived from ``(loc, scale)`` as
    ``loc + scale * Phi_inv(level)``.
    """
    if output_type == "quantile":
        return np.asarray(pred, dtype=np.float64)
    # distribution (gaussian): pred is (B, L, C, 2) = (loc, scale)
    pred = np.asarray(pred, dtype=np.float64)
    loc = pred[..., 0]
    scale = pred[..., 1]
    z = _phi_inv(np.asarray(levels, dtype=np.float64))  # (Q,)
    return loc[..., None] + scale[..., None] * z.reshape(*([1] * loc.ndim), -1)


def collect_prob_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    levels: list[float],
    output_type: str,
    distribution_family: str = "gaussian",
) -> dict[str, float]:
    """Compute probabilistic forecasting metrics.

    Parameters
    ----------
    pred : np.ndarray
        Probabilistic predictions. ``quantile``: ``(B, L, C, Q)`` ordered to
        match ``levels``. ``distribution`` (gaussian): ``(B, L, C, 2)`` with
        ``[..., 0] = loc`` and ``[..., 1] = scale > 0``.
    true : np.ndarray
        Ground-truth targets, shape ``(B, L, C)``.
    levels : list[float]
        Configured quantile levels (sorted ascending).
    output_type : str
        ``"quantile"`` or ``"distribution"``.
    distribution_family : str, optional
        Distribution family when ``output_type == "distribution"`` (default
        ``"gaussian"``).

    Returns
    -------
    dict[str, float]
        Exactly ``{"crps", "wql", "coverage_80", "width_80"}`` as plain floats.

    Notes
    -----
    All four metrics are *lower-is-better* except ``coverage_80``, which is a
    diagnostic ("closer to the nominal 0.8 is better"; no ranking direction is
    enforced). The ``coverage_80`` / ``width_80`` keys are computed from the
    ``0.1`` and ``0.9`` quantile levels (the central 80% band, conventionally
    labelled the 90% prediction interval by its lower/upper deciles).
    """
    true = np.asarray(true, dtype=np.float64)
    levels = list(levels)
    q_pred = _quantile_grid(pred, levels, output_type)  # (B, L, C, Q)
    Q = q_pred.shape[-1]

    # --- CRPS -------------------------------------------------------------
    if output_type == "distribution" and distribution_family == "gaussian":
        # Closed-form Gaussian CRPS.
        pred_arr = np.asarray(pred, dtype=np.float64)
        mu = pred_arr[..., 0]
        sigma = pred_arr[..., 1]
        omega = (true - mu) / sigma
        crps_elem = sigma * (omega * (2.0 * _phi_cdf(omega) - 1.0) + 2.0 * _phi_pdf(omega) - 1.0 / math.sqrt(math.pi))
        crps = float(np.mean(crps_elem))
    else:
        # Quantile (CRPS approximation): (2/Q) * sum_q mean(pinball_q).
        crps = float((2.0 / Q) * sum(np.mean(_pinball(q_pred[..., i], true, levels[i])) for i in range(Q)))

    # --- WQL (GIFT-Eval / GluonTS weighted quantile loss) -----------------
    # GluonTS MeanWeightedSumQuantileLoss / GIFT-Eval WQL is the MEAN over
    # quantiles of the per-quantile weighted loss:
    #   wql = (1/Q) * sum_q ( 2 * sum(pinball_q) / sum|y| )
    # The 1/Q averaging is required so the metric matches externally published
    # WQL numbers; without it the value is exactly Q times too large. For the
    # quantile path this also satisfies the identity wql == crps / mean|y|.
    numerator = sum(2.0 * np.sum(_pinball(q_pred[..., i], true, levels[i])) for i in range(Q))
    denominator = np.sum(np.abs(true))
    wql = float(numerator / (denominator + 1e-12) / Q)

    # --- coverage_80 / width_80 (the [q0.1, q0.9] band) -------------------
    lo_idx = _level_index(levels, 0.1, fallback=0)
    hi_idx = _level_index(levels, 0.9, fallback=len(levels) - 1)
    lo = q_pred[..., lo_idx]
    hi = q_pred[..., hi_idx]
    coverage_80 = float(np.mean((true >= lo) & (true <= hi)))
    width_80 = float(np.mean(hi - lo))

    return {
        "crps": crps,
        "wql": wql,
        "coverage_80": coverage_80,
        "width_80": width_80,
    }


def _level_index(levels: list[float], target: float, fallback: int) -> int:
    """Index of ``target`` in ``levels`` (via ``np.isclose``), else ``fallback``."""
    for i, lvl in enumerate(levels):
        if np.isclose(lvl, target):
            return i
    return fallback


def _prob_metric_guard(name: str):
    """Return a registry placeholder for a probabilistic metric.

    The four probabilistic metrics (``crps``/``wql``/``coverage_80``/
    ``width_80``) are *not* bare ``(pred, true)`` callables — they require the
    quantile levels and the model's ``output_type`` and are computed together by
    :func:`collect_prob_metrics`. They are registered here only so that a name
    that is mapped in ``METRIC_NAME_MAP`` and marked as registered stays
    resolvable via ``METRIC_REGISTRY.get(name)`` (preserving the registry
    invariant). Calling the placeholder directly raises a descriptive error
    pointing to the real call path.
    """

    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(
            f"Probabilistic metric '{name}' is not callable as a bare "
            "(pred, true) metric; it is computed by "
            "tsflab.experiments.evaluation.metrics.collect_prob_metrics("
            "pred, true, levels, output_type, ...)."
        )

    _placeholder.__name__ = name
    return _placeholder


def register() -> None:
    """Register built-in metrics into the registry."""
    METRIC_REGISTRY.register("mae", mae)
    METRIC_REGISTRY.register("mse", mse)
    METRIC_REGISTRY.register("rmse", rmse)
    METRIC_REGISTRY.register("mape", mape)
    METRIC_REGISTRY.register("mspe", mspe)
    METRIC_REGISTRY.register("corr", corr)
    METRIC_REGISTRY.register("rse", rse)
    METRIC_REGISTRY.register("wape", wape)
    METRIC_REGISTRY.register("smape", smape)
    METRIC_REGISTRY.register("mase", mase)
    # Probabilistic metrics: register resolvable placeholders so a mapped +
    # marked name never KeyErrors on METRIC_REGISTRY.get(). The real values are
    # produced by collect_prob_metrics (the registry is not used for eval).
    for _name in ("crps", "wql", "coverage_80", "width_80"):
        METRIC_REGISTRY.register(_name, _prob_metric_guard(_name))

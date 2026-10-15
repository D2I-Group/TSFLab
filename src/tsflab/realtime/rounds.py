"""Open rounds, store forecasts, and score them once the truth is released.

Rounds live in the repository so that every forecast and score is reviewable
evidence::

    apps/web/submissions/realtime/<track>/rounds/<round_id>/
        round.json               # RoundSpec
        forecasts/<model>.json   # ForecastSubmission (one per method)
        scores.json              # RoundScore list, written once the truth exists
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tsflab.core.realtime import ForecastSubmission, RoundScore, RoundSpec
from tsflab.realtime.store import PanelStore
from tsflab.realtime.tracks import TrackSpec

ROUNDS_ROOT = Path("apps") / "web" / "submissions" / "realtime"


def _iso(ts: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(ts).isoformat()


def _local(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    """Attach the track's timezone to a naive panel timestamp."""
    return ts.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)


def _naive(ts: str) -> pd.Timestamp:
    """Panel index value (naive wall time) of a stored ISO timestamp."""
    stamp = pd.Timestamp(ts)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def round_dir(track: str, round_id: str, root: Path = ROUNDS_ROOT) -> Path:
    return root / track / "rounds" / round_id


def list_rounds(track: str, root: Path = ROUNDS_ROOT) -> list[RoundSpec]:
    specs = []
    for path in sorted((root / track / "rounds").glob("*/round.json")):
        specs.append(RoundSpec.model_validate_json(path.read_text(encoding="utf-8")))
    return specs


def load_round(track: str, round_id: str, root: Path = ROUNDS_ROOT) -> RoundSpec:
    return RoundSpec.model_validate_json((round_dir(track, round_id, root) / "round.json").read_text(encoding="utf-8"))


def target_timestamps(track: TrackSpec, cutoff: pd.Timestamp, opened_at: pd.Timestamp) -> list[pd.Timestamp]:
    """Forecast targets: ``horizon`` grid steps starting after the submission window."""
    offset = pd.tseries.frequencies.to_offset(track.freq)
    opened_local = opened_at.tz_convert(track.tz).tz_localize(None)
    earliest = max(cutoff, opened_local + pd.Timedelta(hours=track.submission_hours))
    grid = pd.date_range(cutoff, periods=track.horizon + 10_000, freq=offset)
    future = [ts for ts in grid if ts > earliest]
    return future[: track.horizon]


def open_round(
    track: TrackSpec,
    store: PanelStore,
    *,
    now: datetime | None = None,
    hf_revision: str | None = None,
    root: Path = ROUNDS_ROOT,
    cutoff: pd.Timestamp | None = None,
) -> RoundSpec:
    """Create the round for the store's latest release (idempotent per ISO week).

    ``cutoff`` overrides the release's last timestamp; replays use it to open
    rounds at historical instants on a store that already holds their truth.
    """
    release = store.latest_release()
    if release is None:
        raise RuntimeError(f"track {track.id!r} has no release to open a round on")
    opened = pd.Timestamp(now or datetime.now(UTC))
    opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
    year, week, _ = opened.isocalendar()
    round_id = f"{year}-W{week:02d}"
    directory = round_dir(track.id, round_id, root)
    if (directory / "round.json").is_file():
        return load_round(track.id, round_id, root)
    replay = cutoff is not None
    cutoff = pd.Timestamp(cutoff if replay else release["last_timestamp"])
    panel = store.read(start=cutoff - pd.Timedelta(days=max(28, (track.history_days or 28))), end=cutoff)
    stats = panel.tail(max(track.seq_len * 4, 64))
    mean = stats.mean(skipna=True).fillna(0.0)
    std = stats.std(skipna=True)
    # Floor the scale so dead or near-constant sensors cannot dominate the
    # z-scored error: at least 5% of the typical channel scale.
    floor = max(float(np.nanmedian(std.to_numpy())) * 0.05, 1e-8) if std.notna().any() else 1.0
    std = std.fillna(floor).clip(lower=floor)
    targets = target_timestamps(track, cutoff, opened)
    spec = RoundSpec(
        track=track.id,
        round_id=round_id,
        data_version=f"replay@{release['version']}" if replay else release["version"],
        hf_revision=hf_revision,
        mode=track.mode,
        freq=track.freq,
        cutoff=_iso(_local(cutoff, track.tz)),
        target_timestamps=[_iso(_local(t, track.tz)) for t in targets],
        channels=list(panel.columns),
        seq_len=track.seq_len,
        opened_at=_iso(opened),
        deadline=_iso(_local(targets[0], track.tz) - pd.Timedelta(seconds=1)),
        norm_mean=[float(v) for v in mean.to_numpy()],
        norm_std=[float(v) for v in std.to_numpy()],
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "round.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return spec


def write_forecast(submission: ForecastSubmission, spec: RoundSpec, root: Path = ROUNDS_ROOT) -> Path:
    submission.check_against(spec)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in submission.model)
    path = round_dir(spec.track, spec.round_id, root) / "forecasts" / f"{safe}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(submission.model_dump_json() + "\n", encoding="utf-8")
    return path


def load_forecasts(spec: RoundSpec, root: Path = ROUNDS_ROOT) -> list[ForecastSubmission]:
    directory = round_dir(spec.track, spec.round_id, root) / "forecasts"
    return [
        ForecastSubmission.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))
    ]


def score_round(
    spec: RoundSpec, store: PanelStore, min_coverage: float, root: Path = ROUNDS_ROOT
) -> list[RoundScore] | None:
    """Score every forecast of ``spec``; ``None`` while the truth is incomplete."""
    targets = pd.DatetimeIndex([_naive(t) for t in spec.target_timestamps])
    truth = store.read(start=targets[0], end=targets[-1]).reindex(index=targets, columns=spec.channels)
    observed = ~truth.isna().to_numpy()
    coverage = float(observed.mean())
    if coverage < min_coverage:
        return None
    mean = np.asarray(spec.norm_mean)[None, :]
    std = np.asarray(spec.norm_std)[None, :]
    z_truth = (truth.to_numpy() - mean) / std
    scores = []
    for forecast in load_forecasts(spec, root):
        forecast.check_against(spec)
        z_pred = (np.asarray(forecast.predictions, dtype=float) - mean) / std
        err = (z_pred - z_truth)[observed]
        scores.append(
            RoundScore(
                track=spec.track,
                round_id=spec.round_id,
                model=forecast.model,
                submitter=forecast.submitter,
                mse=float(np.mean(err**2)),
                mae=float(np.mean(np.abs(err))),
                coverage=coverage,
            )
        )
    for rank, score in enumerate(sorted(scores, key=lambda s: s.mse), start=1):
        score.rank = rank
    scores.sort(key=lambda s: s.rank)
    path = round_dir(spec.track, spec.round_id, root) / "scores.json"
    path.write_text(json.dumps([s.model_dump() for s in scores], indent=2) + "\n", encoding="utf-8")
    return scores


def load_scores(track: str, root: Path = ROUNDS_ROOT) -> dict[str, list[RoundScore]]:
    out = {}
    for path in sorted((root / track / "rounds").glob("*/scores.json")):
        out[path.parent.name] = [RoundScore.model_validate(s) for s in json.loads(path.read_text(encoding="utf-8"))]
    return out


def kendall_tau(a: list[str], b: list[str]) -> float | None:
    """Kendall's tau between two rankings over their common methods."""
    common = [m for m in a if m in set(b)]
    if len(common) < 2:
        return None
    pos_b = {m: i for i, m in enumerate(m for m in b if m in set(common))}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if pos_b[common[i]] < pos_b[common[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def track_summary(track: str, root: Path = ROUNDS_ROOT) -> dict:
    """Aggregate scored rounds: mean rank/MSE per method and rank stability."""
    rounds = load_scores(track, root)
    per_method: dict[str, list[RoundScore]] = {}
    for scores in rounds.values():
        for score in scores:
            per_method.setdefault(score.model, []).append(score)
    methods = sorted(
        (
            {
                "model": m,
                "rounds": len(v),
                "mean_mse": float(np.mean([s.mse for s in v])),
                "mean_rank": float(np.mean([s.rank for s in v])),
            }
            for m, v in per_method.items()
        ),
        key=lambda r: (r["mean_rank"], r["mean_mse"]),
    )
    ordered = [[s.model for s in rounds[r]] for r in sorted(rounds)]
    taus = [kendall_tau(x, y) for x, y in zip(ordered, ordered[1:])]
    taus = [t for t in taus if t is not None]
    return {
        "track": track,
        "scored_rounds": sorted(rounds),
        "methods": methods,
        "rank_stability": {"consecutive_kendall_tau": taus, "mean": float(np.mean(taus)) if taus else None},
    }

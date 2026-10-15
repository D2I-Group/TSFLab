"""CSI-300 constituent daily log returns through AKShare (free, no key).

The panel stores ``log(close_t / close_{t-1})`` of forward-adjusted (``qfq``)
closes, so dividends and splits do not appear as jumps and values are
comparable across symbols. AKShare throttles large historical requests, so the
fetcher pulls only the requested range per symbol and pauses between symbols.
"""

from __future__ import annotations

import pandas as pd

from tsflab.realtime.sources.equity import fetch_panel, with_fallback
from tsflab.realtime.tracks import TrackSpec


def _akshare():
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("the stock track needs `pip install akshare`") from exc
    return ak


def constituents() -> list[str]:
    ak = _akshare()
    frame = ak.index_stock_cons_csindex(symbol="000300")
    return sorted({str(code).zfill(6) for code in frame["成分券代码"]})


def _eastmoney(ak, symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str) -> pd.Series:
    history = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    return pd.Series(history["收盘"].to_numpy(dtype=float), index=pd.to_datetime(history["日期"]))


def _sina(ak, symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str) -> pd.Series:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    history = ak.stock_zh_a_daily(
        symbol=prefix + symbol, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust=adjust
    )
    return pd.Series(history["close"].to_numpy(dtype=float), index=pd.to_datetime(history["date"]))


_PREFERRED = [_eastmoney, _sina]


def daily_close(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str = "qfq", attempts: int = 4
) -> pd.Series:
    """Adjusted daily closes, falling back across vendors (see ``with_fallback``)."""
    return with_fallback(_PREFERRED, _akshare(), symbol, start, end, adjust, attempts=attempts, label=symbol)


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    adjust = track.source.get("adjust", "qfq")
    return fetch_panel(
        track, start, end, channels or constituents(), lambda symbol, lo, hi: daily_close(symbol, lo, hi, adjust)
    )

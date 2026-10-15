"""NASDAQ-100 constituent daily log returns (free, no key).

Constituents come from the Nasdaq list endpoint. Closes come from the Nasdaq
historical endpoint first, then the Yahoo chart API; both are split- but not
dividend-adjusted (``us_prices``), so a day served by either is identical. Sina
through AKShare (forward-adjusted, dividend-adjusted) stays as the last resort
and can differ from the others by the dividend yield on ex-dividend days.
Symbols are stored with ``.`` replaced by ``-`` (e.g. ``BRK-B``).
"""

from __future__ import annotations

import pandas as pd

from tsflab.realtime.sources.equity import fetch_panel
from tsflab.realtime.sources.us_prices import close_with_fallback, nasdaq_close, nasdaq_get, yahoo_close
from tsflab.realtime.tracks import TrackSpec


def constituents() -> list[str]:
    rows = nasdaq_get("quote/list-type/nasdaq100")["data"]["rows"]
    return sorted({row["symbol"].strip().upper().replace(".", "-") for row in rows})


def _sina(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("the Sina fallback needs `pip install akshare`") from exc
    history = ak.stock_us_daily(symbol=symbol.replace("-", "."), adjust="qfq")
    close = pd.Series(history["close"].to_numpy(dtype=float), index=pd.to_datetime(history["date"]))
    return close.loc[start:end]


_PREFERRED = [nasdaq_close, yahoo_close, _sina]


def daily_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return close_with_fallback(_PREFERRED, symbol, start, end)


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    return fetch_panel(track, start, end, channels or constituents(), daily_close)

"""Daily US equity closes from public endpoints (no key): Nasdaq API and Yahoo chart API.

Both vendors return split-adjusted but not dividend-adjusted closes, so the two
can serve the same track interchangeably: a day served by either one equals
the traded close after splits. Tickers use the Yahoo convention (``BRK-B``);
the Nasdaq API takes ``BRK.B``.

Fetchers raise on an empty response so that ``equity.with_fallback`` moves on to
the next vendor instead of silently dropping a symbol.
"""

from __future__ import annotations

import pandas as pd

from tsflab.realtime.sources.equity import with_fallback

_NASDAQ = "https://api.nasdaq.com/api"
_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart"
_HEADERS = {"User-Agent": "Mozilla/5.0 (TSFLab real-time track)", "Accept": "application/json"}


def nasdaq_get(path: str, **params) -> dict:
    import requests

    response = requests.get(f"{_NASDAQ}/{path}", params=params, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("data"):
        raise RuntimeError(f"Nasdaq API returned no data for {path}: {payload.get('status')}")
    return payload["data"]


def parse_nasdaq_history(data: dict) -> pd.Series:
    """``/quote/<symbol>/historical`` payload -> closes indexed by trading date."""
    rows = (data.get("tradesTable") or {}).get("rows") or []
    if not rows:
        raise RuntimeError("Nasdaq API returned no rows")
    values = [float(str(row["close"]).replace("$", "").replace(",", "")) for row in rows]
    series = pd.Series(values, index=pd.to_datetime([row["date"] for row in rows], format="%m/%d/%Y"))
    return series.sort_index()  # the endpoint lists newest first


def nasdaq_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    # The endpoint returns nothing for a short window that ends long before today,
    # so callers always request up to the present (updates and bootstraps do).
    data = nasdaq_get(
        f"quote/{symbol.replace('-', '.')}/historical",
        assetclass="stocks",
        fromdate=f"{start:%Y-%m-%d}",
        todate=f"{end:%Y-%m-%d}",
        limit=9999,
    )
    return parse_nasdaq_history(data)


def parse_yahoo_chart(payload: dict, now: pd.Timestamp | None = None) -> pd.Series:
    """Yahoo chart payload -> split-adjusted closes indexed by exchange-local date.

    The bar of a session that is still open is dropped: only settled closes enter a panel.
    """
    result = (payload.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo chart returned no result: {(payload.get('chart') or {}).get('error')}")
    result = result[0]
    zone = result["meta"].get("exchangeTimezoneName", "America/New_York")
    stamps = pd.to_datetime(result.get("timestamp") or [], unit="s", utc=True).tz_convert(zone)
    close = pd.Series(result["indicators"]["quote"][0]["close"], index=stamps, dtype="float64").dropna()
    if close.empty:
        raise RuntimeError("Yahoo chart returned no closes")
    close.index = close.index.tz_localize(None).normalize()
    now = (now or pd.Timestamp.now(tz=zone)).tz_convert(zone)
    if now.hour < 17:  # the session of ``now``'s date has not settled yet
        close = close[close.index < now.tz_localize(None).normalize()]
    return close[~close.index.duplicated(keep="last")]


def yahoo_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import requests

    response = requests.get(
        f"{_YAHOO}/{symbol.replace('.', '-')}",
        headers=_HEADERS,
        timeout=30,
        params={
            "period1": int(start.tz_localize("UTC").timestamp()),
            "period2": int((end + pd.Timedelta(days=2)).tz_localize("UTC").timestamp()),
            "interval": "1d",
            "events": "div,splits",
        },
    )
    response.raise_for_status()
    return parse_yahoo_chart(response.json())


def close_with_fallback(providers: list, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Closes in ``[start, end]``; a symbol no vendor can serve yields an empty series."""
    try:
        return with_fallback(providers, symbol, start, end, attempts=2, label=symbol).loc[start:end]
    except RuntimeError:
        return pd.Series(dtype="float64")

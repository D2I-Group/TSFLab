"""S&P 500 constituent daily log returns (free, no key).

Constituents come from the public ``datasets/s-and-p-500-companies`` CSV (kept
current from Wikipedia, ODC-PDDL) with the Wikipedia list as fallback. Closes
come from the Nasdaq historical endpoint, then the Yahoo chart API (see
``us_prices``); both are split-adjusted, not dividend-adjusted.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from tsflab.realtime.sources.equity import fetch_panel
from tsflab.realtime.sources.us_prices import close_with_fallback, nasdaq_close, yahoo_close
from tsflab.realtime.tracks import TrackSpec

CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
WIKI_URL = (
    "https://en.wikipedia.org/w/api.php?action=parse&page=List_of_S%26P_500_companies"
    "&prop=wikitext&section=1&format=json"
)
_HEADERS = {"User-Agent": "TSFLab real-time track (research)"}


def _clean(symbols) -> list[str]:
    return sorted({str(s).strip().upper().replace(".", "-") for s in symbols if str(s).strip()})


def parse_constituents_csv(text: str) -> list[str]:
    return _clean(pd.read_csv(io.StringIO(text))["Symbol"])


def parse_constituents_wikitext(wikitext: str) -> list[str]:
    return _clean(re.findall(r"\{\{\s*(?:Nyse|Nasdaq|Cboe|Bats)\w*Symbol\s*\|\s*([A-Za-z.\-]+)", wikitext))


def constituents() -> list[str]:
    import requests

    try:
        response = requests.get(CSV_URL, headers=_HEADERS, timeout=30)
        response.raise_for_status()
        symbols = parse_constituents_csv(response.text)
    except Exception:  # fall back to the Wikipedia list
        response = requests.get(WIKI_URL, headers=_HEADERS, timeout=30)
        response.raise_for_status()
        symbols = parse_constituents_wikitext(response.json()["parse"]["wikitext"]["*"])
    if len(symbols) < 450:
        raise RuntimeError(f"S&P 500 constituent list looks truncated ({len(symbols)} symbols)")
    return symbols


_PREFERRED = [nasdaq_close, yahoo_close]


def daily_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return close_with_fallback(_PREFERRED, symbol, start, end)


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    return fetch_panel(track, start, end, channels or constituents(), daily_close)

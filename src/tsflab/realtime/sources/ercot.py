"""ERCOT hourly load by weather zone from public files (no key).

History: the yearly ``Native_Load_<year>.zip`` archives on ercot.com (an xlsx
read with the standard library, so no ``openpyxl`` is needed; the current-year
file is refreshed monthly). Updates: the Market Information System report
NP6-345-CD "Actual System Load by Weather Zone", whose daily files stay listed
for roughly a month. Both give hour-ending values in ERCOT local clock time
(America/Chicago); a stamp is moved to the start of the hour it covers, and the
repeated hour of the autumn clock change is averaged. The ERCOT Public API
(api.ercot.com, registration + subscription key) holds the same reports with
deeper history but is not needed here.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

from tsflab.realtime.tracks import TrackSpec

MIS_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
MIS_DOWNLOAD = "https://www.ercot.com/misdownload/servlets/mirDownload"
ARCHIVE_PAGE = "https://www.ercot.com/gridinfo/load/load_hist"
_HEADERS = {"User-Agent": "Mozilla/5.0 (TSFLab real-time track)"}
ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]
_RENAME = {"FAR_WEST": "FWEST", "NORTH_C": "NCENT", "SOUTHERN": "SOUTH", "SOUTH_C": "SCENT"}
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _hour_ending(stamps: pd.Series) -> pd.DatetimeIndex:
    """``MM/DD/YYYY HH:00`` (``24:00`` = end of day) -> start of the covered hour."""
    text = stamps.astype(str).str.strip()
    day = pd.to_datetime(text.str[:10], format="%m/%d/%Y")
    hour = text.str[11:13].astype(int)
    return pd.DatetimeIndex(day + pd.to_timedelta(hour - 1, unit="h"))


def _collapse(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.groupby(level=0).mean().sort_index()  # repeated autumn hour -> mean
    frame.index.name = None
    return frame[ZONES]


def read_xlsx_rows(payload: bytes) -> list[list[str]]:
    """Cell text of the first worksheet of an xlsx file (standard library only)."""
    book = zipfile.ZipFile(io.BytesIO(payload))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in book.namelist():
        for item in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in item.iter(f"{{{_NS['m']}}}t")))
    sheet = sorted(n for n in book.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))[0]
    rows = []
    for row in ET.fromstring(book.read(sheet)).iter(f"{{{_NS['m']}}}row"):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", _NS):
            column = 0
            for char in re.match(r"[A-Z]+", cell.get("r")).group():
                column = column * 26 + ord(char) - 64
            kind, value = cell.get("t"), cell.find("m:v", _NS)
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in cell.iter(f"{{{_NS['m']}}}t"))
            elif value is None:
                text = ""
            else:
                text = shared[int(value.text)] if kind == "s" else value.text
            cells[column - 1] = text
        rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
    return rows


def parse_native_load(payload: bytes) -> pd.DataFrame:
    """A ``Native_Load_<year>.zip`` (or bare xlsx) -> hourly load per weather zone, local time."""
    if payload[:4] == b"PK\x03\x04" and any(
        n.endswith(".xlsx") for n in zipfile.ZipFile(io.BytesIO(payload)).namelist()
    ):
        archive = zipfile.ZipFile(io.BytesIO(payload))
        payload = archive.read(next(n for n in archive.namelist() if n.endswith(".xlsx")))
    rows = read_xlsx_rows(payload)
    header = [h.strip() for h in rows[0]]
    body = pd.DataFrame([r + [""] * (len(header) - len(r)) for r in rows[1:] if r], columns=header)
    # 2019/2020 archives spell the stamp column ``HourEnding``, later ones ``Hour Ending``.
    stamp_col = next((h for h in header if h.replace(" ", "").lower() == "hourending"), None)
    if stamp_col is None:
        raise ValueError(f"Native_Load sheet has no hour-ending column; header: {header}")
    body = body[body[stamp_col].str.strip() != ""]
    frame = body[ZONES].apply(pd.to_numeric, errors="coerce")
    frame.index = _hour_ending(body[stamp_col].map(_stamp_text))
    return _collapse(frame)


_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _stamp_text(value: str) -> str:
    """Pass ``MM/DD/YYYY HH:00`` through; turn an Excel serial date into that form.

    A few archive cells (e.g. 2022-12-01 01:00 in ``Native_Load_2022``) hold the
    stamp as a numeric serial (``44896.041666``) instead of text. Serials cannot
    say ``24:00``; midnight is written as ``00:00`` of the next day, which
    ``_hour_ending`` maps to the same covered hour.
    """
    text = str(value).strip()
    try:
        serial = float(text)
    except ValueError:
        return text
    stamp = (_EXCEL_EPOCH + pd.to_timedelta(serial, unit="D")).round("h")
    return stamp.strftime("%m/%d/%Y %H:00")


def parse_weather_zone_csv(text: str) -> pd.DataFrame:
    """NP6-345-CD csv (``OperDay,HourEnding,COAST,...``) -> hourly load per weather zone."""
    raw = pd.read_csv(io.StringIO(text)).rename(columns=_RENAME)
    stamps = raw["OperDay"].astype(str) + " " + raw["HourEnding"].astype(str)
    frame = raw[ZONES].apply(pd.to_numeric, errors="coerce")
    frame.index = _hour_ending(stamps)
    return _collapse(frame)


def _get(url: str, **params):
    import requests

    response = requests.get(url, params=params, headers=_HEADERS, timeout=120)
    response.raise_for_status()
    return response


def archive_urls() -> dict[int, str]:
    links = re.findall(r'href="([^"]*[Nn]ative_[Ll]oad_(\d{4})\.zip)"', _get(ARCHIVE_PAGE).text)
    return {int(year): url for url, year in links}


def recent_daily_texts(since: pd.Timestamp) -> list[str]:
    listing = _get(MIS_LIST, reportTypeId=13101).json()["ListDocsByRptTypeRes"]["DocumentList"]
    texts = []
    for entry in (item["Document"] for item in listing):
        published = pd.Timestamp(entry["PublishDate"]).tz_localize(None)
        if "csv" not in entry["FriendlyName"].lower() or published < since.normalize():
            continue
        archive = zipfile.ZipFile(io.BytesIO(_get(MIS_DOWNLOAD, doclookupId=entry["DocID"]).content))
        texts.append(archive.read(archive.namelist()[0]).decode("utf-8", errors="replace"))
    return texts


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    frames = []
    urls = archive_urls()
    for year in range(start.year, end.year + 1):
        if year in urls:
            frames.append(parse_native_load(_get(urls[year]).content))
    # the archive trails by a month: the rolling daily reports cover the rest
    frames.extend(parse_weather_zone_csv(text) for text in recent_daily_texts(start))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel = panel[~panel.index.duplicated(keep="first")].sort_index()  # archive wins on overlap
    return panel.loc[start:end]

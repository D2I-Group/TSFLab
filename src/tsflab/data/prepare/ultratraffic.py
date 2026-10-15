"""Convert the UltraTraffic_CL archive into a local parquet store.

UltraTraffic_CL (distributed inside ``TrafficCL.zip``) holds hourly total flow
per Caltrans PeMS station, one CSV per region and year (2003-2023), in two
variants: ``<R>_Static/<year>.csv`` (all stations observed that year) and
``<R>_CL/<year>_{common,added}.csv`` (stations retained from the previous year
and stations new that year, for continual-learning settings). The CL full-year
file is byte-identical to the Static one and is not stored twice.

Layout written under ``--out`` (default ``dataset/ultratraffic``)::

    <REGION>/static/<year>.parquet
    <REGION>/cl/<year>_common.parquet
    <REGION>/cl/<year>_added.parquet
    manifest.json

The archive's ``PEMS_NC`` directory is byte-identical to ``PEMS_SAC`` (both are
Caltrans District 3); it is recorded as a duplicate and skipped.

    uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

# Region directory -> Caltrans district (from station-id prefixes).
REGIONS = {"PEMS_BA": 4, "PEMS_LA": 7, "PEMS_SAC": 3, "PEMS_SB": 8}
DUPLICATES = {"PEMS_NC": "PEMS_SAC"}
_MEMBER = re.compile(r"UltraTraffic_CL/(PEMS_\w+)/\w+_(Static|CL)/(\d{4})(?:_(common|added))?\.csv$")


def _inner_archive(path: Path) -> zipfile.ZipFile:
    """Open UltraTraffic_CL.zip directly or through the outer TrafficCL.zip."""
    outer = zipfile.ZipFile(path)
    names = outer.namelist()
    if any(n.startswith("UltraTraffic_CL/") for n in names):
        return outer
    inner = next(n for n in names if n.endswith("UltraTraffic_CL.zip"))
    return zipfile.ZipFile(outer.open(inner))


def convert(archive: Path, out: Path, regions: list[str] | None = None) -> dict:
    source = _inner_archive(archive)
    manifest = {"source": archive.name, "regions": {}, "duplicates": DUPLICATES}
    wanted = set(regions or REGIONS)
    for name in sorted(source.namelist()):
        match = _MEMBER.search(name)
        if not match:
            continue
        region, variant, year, part = match.groups()
        if region not in wanted or region in DUPLICATES:
            continue
        if variant == "CL" and part is None:
            continue  # identical to the Static full-year file
        target = (
            out
            / region
            / ("static" if variant == "Static" else "cl")
            / (f"{year}.parquet" if part is None else f"{year}_{part}.parquet")
        )
        entry = manifest["regions"].setdefault(region, {"district": REGIONS[region], "files": {}})
        key = str(target.relative_to(out / region))
        if target.is_file():
            entry["files"][key] = json.loads((target.with_suffix(".json")).read_text())
            continue
        payload = source.read(name)
        frame = pd.read_csv(io.BytesIO(payload), index_col="date", parse_dates=True)
        frame.columns = [str(c) for c in frame.columns]
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.astype("float32").to_parquet(target)
        facts = {
            "rows": int(len(frame)),
            "stations": int(frame.shape[1]),
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "missing": round(float(frame.isna().mean().mean()), 4),
            "csv_sha256": hashlib.sha256(payload).hexdigest(),
        }
        target.with_suffix(".json").write_text(json.dumps(facts, indent=1))
        entry["files"][key] = facts
        print(f"{region} {key}: {facts['stations']} stations, {facts['rows']} hours", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tsf data prepare --from ultratraffic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive", type=Path, required=True, help="TrafficCL.zip or UltraTraffic_CL.zip")
    parser.add_argument("--out", type=Path, default=Path("dataset") / "ultratraffic")
    parser.add_argument("--regions", nargs="*", help="subset of " + ", ".join(REGIONS))
    args = parser.parse_args(argv)
    convert(args.archive, args.out, args.regions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

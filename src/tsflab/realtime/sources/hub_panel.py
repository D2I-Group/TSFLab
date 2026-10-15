"""Bootstrap a track from a wide CSV already published on the Hugging Face Hub."""

from __future__ import annotations

import pandas as pd

from tsflab.realtime.tracks import TrackSpec
from tsflab.release.hub import fetch


def load(track: TrackSpec) -> pd.DataFrame:
    path = fetch(track.bootstrap["uri"])
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.columns = [str(c).zfill(6) if str(c).isdigit() else str(c) for c in frame.columns]
    return frame.sort_index()

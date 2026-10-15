"""Runtime specification for STELLA."""

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stella.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``enc_in`` is the number of stations. ``coords_path`` names a local
    ``.npy`` array ``[stations, 3]`` of geographic coordinates (for example
    latitude, longitude, altitude) in channel order; without it the spatial
    embedding is a learnable per-station table (the paper's RPE ablation).
    ``adj_mx``/``num_nodes`` injected by the runner are not used.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=32, ge=1)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    coords_path: str | None = None


def load_coordinates(path: str | None) -> np.ndarray | None:
    """Read station coordinates from a local ``.npy`` file; never downloads."""
    if path is None:
        return None
    location = Path(path).expanduser()
    if location.suffix != ".npy" or not location.is_file():
        raise ValueError(f"coords_path must be an existing local .npy file: {path}")
    return np.load(location, allow_pickle=False)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        coordinates=load_coordinates(params.get("coords_path")),
    )


SPEC = ModelSpec(
    name="STELLA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_stella.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("marks",),
    contract_task={"seq_len": 48, "pred_len": 24, "label_len": 0},
)

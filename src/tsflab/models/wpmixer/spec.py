"""Model specification for WPMixer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.wpmixer.model import Model

from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    dropout: float = 0.1
    tfactor: int = 5
    dfactor: int = 5
    wavelet: str = "db2"
    level: int = 1
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    no_decomposition: bool = False

    @model_validator(mode="after")
    def _wavelet_filters(self) -> "ModelParameterConfig":
        # The wavelet is only built when the decomposition is on.
        if not self.no_decomposition and (self.wavelet not in ("haar", "db1", "db2") or self.level < 1):
            raise ValueError("wavelet must be haar/db1/db2 and level positive")
        return self


def build_model(cfg, params):
    """Construct WPMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        tfactor=params["tfactor"],
        dfactor=params["dfactor"],
        wavelet=params["wavelet"],
        level=params["level"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        no_decomposition=bool(params["no_decomposition"]),
    )


SPEC = ModelSpec(
    name="WPMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

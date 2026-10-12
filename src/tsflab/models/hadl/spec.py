"""Runtime specification for HADL."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.hadl.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``enable_*`` reproduce the paper's ablations (Sec. 6, Appendix C);
    ``regularization_rate`` scales the official L1 output penalty (0 disables it).
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    rank: int = Field(default=50, gt=0)
    bias: bool = True
    individual: bool = False
    enable_haar: bool = True
    enable_dct: bool = True
    enable_lowrank: bool = True
    regularization_rate: float = Field(default=0.1, ge=0.0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="HADL",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_hadl.toml",
    capabilities=frozenset({"time-series"}),
    components=("haar_dwt1d",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

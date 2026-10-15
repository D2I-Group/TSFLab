"""Runtime specification for FreqMoE."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.freqmoe.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    expert_num: int = Field(default=4, ge=1)
    freq_num_blocks: int = Field(default=1, ge=1)
    dropout_freq: float = 0.1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        expert_num=params["expert_num"],
        freq_num_blocks=params["freq_num_blocks"],
        dropout_freq=params["dropout_freq"],
    )


SPEC = ModelSpec(
    name="FreqMoE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_freqmoe.toml",
    capabilities=frozenset({"time-series"}),
    components=("freq_band_moe",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

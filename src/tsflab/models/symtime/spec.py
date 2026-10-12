"""Runtime specification for SymTime."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.symtime.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    patch_len: int = 16
    num_layers: int = 2
    num_heads: int = 4
    trend_kernel: int = 25
    dropout: float = 0.1


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        trend_kernel=params["trend_kernel"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="SymTime",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

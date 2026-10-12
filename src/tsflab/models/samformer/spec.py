"""Runtime specification for SAMformer."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.samformer.model import Model

"""Validated parameters for SAMformer."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    hid_dim: int = 16
    rho: float = 0.5
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        hid_dim=params["hid_dim"],
        rho=params["rho"],
        use_revin=params["use_revin"],
    )


def training_objective(model, batch, criterion):
    """Sharpness-aware minimization of the configured criterion (paper section 4)."""
    return model.sharpness_aware_objective(batch, criterion)


SPEC = ModelSpec(
    name="SAMformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_samformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("channel_wise_linear", "revin", "sharpness_aware"),
    training_objective=training_objective,
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

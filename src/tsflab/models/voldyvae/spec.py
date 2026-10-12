"""Runtime specification for VolDyVAE."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.voldyvae.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    patch_len: int = Field(default=24, ge=1)
    dynamic_dim: int = Field(default=128, ge=1)
    hidden_layers: int = Field(default=3, ge=2)
    hidden_dim: int = Field(default=256, ge=1)
    vol_hidden_dim: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    beta: float = Field(default=0.01, ge=0.0)
    num_samples: int = Field(default=100, ge=2)
    xi: float = Field(default=1e-6, gt=0.0)
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    options = dict(params)
    levels = options.pop("quantile_levels", None) or list(cfg.evaluation.quantile_levels)
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        quantile_levels=levels,
        **options,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Replace the configured loss by Eq. (8): reconstruction NLL + forecast NLL + beta * KL."""
    del criterion
    start = -1 if batch.features == "MS" else 0
    return None, model.training_loss(batch.x, batch.target, target_start=start)


SPEC = ModelSpec(
    name="VolDyVAE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_voldyvae.toml",
    capabilities=frozenset({"quantile-output", "time-series"}),
    components=("empirical_quantiles", "revin"),
    contract_task={"seq_len": 96, "pred_len": 192, "label_len": 0},
    training_objective=training_objective,
)

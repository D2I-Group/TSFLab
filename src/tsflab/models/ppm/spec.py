"""Runtime specification for PPM."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.ppm.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=256, ge=1)
    d_ff: int = Field(default=512, ge=1)
    num_samples: int = Field(default=100, ge=2)
    bandwidth: float = Field(default=0.3, gt=0.0)
    alpha: float = Field(default=0.1, ge=0.0)
    log_floor: float = -25.0
    eps: float = Field(default=1e-6, gt=0.0)
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
    """Replace the configured loss by Eq. (10) on ``num_samples`` training draws."""
    del criterion
    samples = model.sample(batch.x)  # [B, K, H, C]
    start = -1 if batch.features == "MS" else 0
    aligned = samples[:, :, -batch.pred_len :, start:]
    return None, model.training_loss(aligned, batch.target)


SPEC = ModelSpec(
    name="PPM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_ppm.toml",
    capabilities=frozenset({"quantile-output", "time-series"}),
    components=("empirical_quantiles",),
    contract_task={"seq_len": 96, "pred_len": 192, "label_len": 0},
    training_objective=training_objective,
)

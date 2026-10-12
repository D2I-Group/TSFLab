"""Runtime specification for StaTS."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.stats.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    hidden: int = Field(default=256, gt=0)
    bands: int = Field(default=2, gt=0)
    diffusion_steps: int = Field(default=50, ge=2)
    beta_start: float = Field(default=1e-5, gt=0.0, lt=1.0)
    beta_end: float = Field(default=0.1, gt=0.0, lt=1.0)
    schedule_epochs: int = Field(default=3, ge=0)
    stage_lr: float = Field(default=1e-3, gt=0.0)
    stage_weight_decay: float = Field(default=5e-4, ge=0.0)
    stage_grad_clip: float = Field(default=5.0, gt=0.0)
    lambda_smooth: float = Field(default=5.0, ge=0.0)
    lambda_init: float = Field(default=0.5, ge=0.0)
    lambda_end: float = Field(default=0.5, ge=0.0)
    lambda_bar: float = Field(default=5e-3, ge=0.0)
    lambda_prog: float = Field(default=0.5, ge=0.0)
    lambda_obj: float = Field(default=0.01, ge=0.0)
    num_samples: int = Field(default=100, ge=2)
    sample_batch_size: int = Field(default=100, gt=0)
    quantile_levels: list[float] | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must not exceed beta_end")
        return self


def build_model(cfg, params):
    options = dict(params)
    levels = options.pop("quantile_levels") or list(cfg.evaluation.quantile_levels)
    if options["bands"] > cfg.task.seq_len // 2 + 1:
        raise ValueError("bands must not exceed the number of rFFT bins of seq_len")
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, quantile_levels=levels, **options
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Stage II (Sec. 3.2): fit FGD with ``L_obj`` under the frozen learned schedule.

    The diffusion is multivariate over every input channel, so the full future is
    the target (the runner selects the ``MS`` target channel at evaluation).
    """
    future = batch.y[:, -batch.pred_len :, :]
    return None, model.denoiser_loss(batch.x, future)


SPEC = ModelSpec(
    name="StaTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_stats.toml",
    capabilities=frozenset({"quantile-output", "time-series", "pretraining-stage"}),
    components=("empirical_quantiles", "instance_norm"),
    contract_task={"seq_len": 96, "label_len": 48, "pred_len": 24},
    training_objective=training_objective,
)

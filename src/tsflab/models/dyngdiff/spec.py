"""Runtime specification for DynGDiff."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.dyngdiff.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    hidden_dim: int = Field(default=64, gt=0)
    num_blocks: int = Field(default=3, gt=0)
    step_embedding_dim: int = Field(default=128, ge=4)
    state_size: int = Field(default=128, ge=2)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    diffusion_steps: int = Field(default=100, ge=2)
    beta_start: float = Field(default=1e-4, gt=0.0, lt=1.0)
    beta_end: float = Field(default=0.1, gt=0.0, lt=1.0)
    policy_hidden: int = Field(default=64, gt=0)
    policy_step_dim: int = Field(default=32, ge=4)
    guidance_scale: float = Field(default=3.0, ge=0.0)
    num_samples: int = Field(default=100, ge=2)
    sample_batch_size: int = Field(default=100, gt=0)
    backbone_epochs: int = Field(default=300, ge=0)
    backbone_batches_per_epoch: int = Field(default=100, gt=0)
    backbone_lr: float = Field(default=1e-3, gt=0.0)
    backbone_grad_clip: float = Field(default=1.0, ge=0.0)
    backbone_lr_patience: int = Field(default=10, ge=0)
    quantile_levels: list[float] | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.state_size % 2:
            raise ValueError("state_size must be even")
        if self.step_embedding_dim % 2 or self.policy_step_dim % 2:
            raise ValueError("step embedding sizes must be even")
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must not exceed beta_end")
        return self


def build_model(cfg, params):
    options = dict(params)
    levels = options.pop("quantile_levels") or list(cfg.evaluation.quantile_levels)
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, quantile_levels=levels, **options)


def training_objective(model, batch: TrainingBatch, criterion):
    """Stage 2: fit the state-aware policy network (Eq. 10) against the frozen backbone.

    The backbone is trained and frozen by ``pretrain`` (stage 1). The window is
    multivariate over every input channel, so the full future is used (the runner
    selects the ``MS`` target channel at evaluation). No forecast is produced.
    """
    future = batch.y[:, -batch.pred_len :, :]
    return None, model.policy_loss(batch.x, future)


SPEC = ModelSpec(
    name="DynGDiff",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_dyngdiff.toml",
    capabilities=frozenset({"quantile-output", "time-series", "pretraining-stage"}),
    components=("empirical_quantiles",),
    contract_task={"seq_len": 96, "label_len": 48, "pred_len": 24},
    training_objective=training_objective,
)

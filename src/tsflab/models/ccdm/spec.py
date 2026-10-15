"""Runtime specification for CCDM."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.ccdm.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    cont_hidden_dim: int = Field(default=128, gt=0)
    pred_hidden_dim: int = Field(default=128, gt=0)
    step_hidden_dim: int = Field(default=128, gt=0)
    encoder_layers: int = Field(default=2, ge=1)
    decoder_layers: int = Field(default=1, ge=1)
    n_depth: int = Field(default=2, ge=0)
    n_heads: int = Field(default=8, gt=0)
    attn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    mlp_ratio: float = Field(default=1.0, gt=0.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    diffusion_steps: int = Field(default=50, ge=1)
    beta_schedule_kind: Literal["quad", "linear", "cosine"] = "quad"
    beta_start: float = Field(default=1e-4, gt=0.0, lt=1.0)
    beta_end: float = Field(default=0.5, gt=0.0, lt=1.0)
    window_norm: bool = True
    contrast_weight: float = Field(default=1e-3, ge=0.0)
    n_negatives: int = Field(default=64, ge=0)
    temperature: float = Field(default=0.1, gt=0.0)
    shuffle_patch_len: int = Field(default=8, gt=0)
    contrast_form: Literal["regression", "similarity"] = "regression"
    num_samples: int = Field(default=100, ge=2)
    sample_batch_size: int = Field(default=100, gt=0)
    quantile_levels: list[float] | None = None

    @model_validator(mode="after")
    def _check(self):
        if (self.cont_hidden_dim + self.pred_hidden_dim) % self.n_heads:
            raise ValueError("cont_hidden_dim + pred_hidden_dim must be divisible by n_heads")
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must not exceed beta_end")
        return self


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
    """Replace the configured loss by the CCDM objective (Eq. 6, Algorithm 1).

    The diffusion is joint over every input channel, so the whole future window is
    the target (the runner selects the ``MS`` target channel at evaluation). No
    forecast is produced by the same pass.
    """
    future = batch.y[:, -batch.pred_len :, :]
    return None, model.training_loss(batch.x, future)


SPEC = ModelSpec(
    name="CCDM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_ccdm.toml",
    capabilities=frozenset({"quantile-output", "time-series"}),
    components=("empirical_quantiles",),
    contract_task={"seq_len": 48, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)

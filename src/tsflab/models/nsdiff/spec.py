"""Runtime specification for NsDiff."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.nsdiff.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_layers: int = Field(default=1, gt=0)
    d_ff: int = Field(default=1024, gt=0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    p_hidden_dims: list[int] = Field(default_factory=lambda: [64, 64], min_length=1)
    variance_hidden: int = Field(default=512, gt=0)
    rolling_length: int = Field(default=96, gt=0)
    denoiser_hidden: int = Field(default=128, gt=0)
    diffusion_steps: int = Field(default=20, ge=2)
    beta_start: float = Field(default=1e-4, gt=0.0, lt=1.0)
    beta_end: float = Field(default=0.01, gt=0.0, lt=1.0)
    num_samples: int = Field(default=100, ge=2)
    sample_batch_size: int = Field(default=10, gt=0)
    quantile_levels: list[float] | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if any(width <= 0 for width in self.p_hidden_dims):
            raise ValueError("p_hidden_dims must be positive")
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must not exceed beta_end")
        return self


def build_model(cfg, params):
    options = dict(params)
    levels = options.pop("quantile_levels") or list(cfg.evaluation.quantile_levels)
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        quantile_levels=levels,
        **options,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Replace the configured loss by the joint NsDiff objective (Algorithm 1, Eq. 13).

    The diffusion is multivariate over every input channel, so the full future window
    is the target (the runner selects the ``MS`` target channel at evaluation). No
    forecast is produced by the same pass.
    """
    future = batch.y[:, -batch.pred_len :, :]
    return None, model.training_loss(batch.x, batch.x_mark, batch.y_mark, future)


SPEC = ModelSpec(
    name="NsDiff",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_nsdiff.toml",
    capabilities=frozenset({"quantile-output", "time-series"}),
    components=(
        "embed",
        "empirical_quantiles",
        "masking",
        "self_attention_family",
        "transformer_encdec",
    ),
    contract_task={"seq_len": 168, "label_len": 84, "pred_len": 24},
    training_objective=training_objective,
)

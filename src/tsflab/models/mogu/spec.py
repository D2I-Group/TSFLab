"""Runtime specification for MoGU."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.mogu.model import Model, mogu_loss, precision_weights


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_experts: int = Field(default=3, gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    head_type: Literal["mlp", "linear"] = "mlp"
    use_norm: bool = True
    denorm_variance: bool = False
    nll_eps: float = Field(default=1e-6, gt=0.0)

    @model_validator(mode="after")
    def _heads_divide_width(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (11): precision-weighted sum of the experts' Gaussian NLLs.

    The configured ``nll_gaussian`` criterion scores the combined forecast on
    validation and test only; training follows the paper's MoGU loss.
    """
    del criterion
    means, variances = model.expert_outputs(batch.x)
    # Experts on the last axis so the shared horizon/target-channel alignment applies.
    means = batch.align(means.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    variances = batch.align(variances.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    loss = mogu_loss(means, variances, precision_weights(variances), batch.target, model.nll_eps)
    # The objective protocol returns a point forecast or None; the combined
    # forecast is a distribution, so the trainer re-runs forward when needed.
    return None, loss


SPEC = ModelSpec(
    name="MoGU",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mogu.toml",
    capabilities=frozenset({"distribution-output", "time-series"}),
    components=("embed", "instance_norm", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)

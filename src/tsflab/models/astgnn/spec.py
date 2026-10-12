"""Runtime specification for ASTGNN."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.astgnn.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params`` (``adj_mx`` is runner-injected)."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    n_heads: int = Field(default=8, ge=1)
    num_layers: int = Field(default=4, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    smooth_layer_num: int = Field(default=0, ge=0)
    teacher_forcing: bool = True

    @model_validator(mode="after")
    def _widths(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model % 2:
            raise ValueError("d_model must be even for the sinusoidal position table")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        adj_mx=params.get("adj_mx"),
        **values,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Teacher-forced decoding (official first stage) or autoregressive decoding (fine-tune stage)."""
    target = batch.target
    if model.teacher_forcing and target.shape[-1] == model.num_nodes:
        forecast = model.teacher_forced(batch.x, target)
    else:
        forecast = batch.forecast(model)
    return forecast, criterion(batch.align(forecast), target)


SPEC = ModelSpec(
    name="ASTGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset({"spatiotemporal"}),
    components=("adj_norm", "embed"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    requires_graph=True,
    training_objective=training_objective,
)

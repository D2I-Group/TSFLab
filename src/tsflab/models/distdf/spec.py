"""Model specification for DistDF."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.distdf.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    gamma: float = Field(default=0.1, ge=0, le=1)
    covariance_eps: float = Field(default=1e-5, gt=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["gamma"],
        params["covariance_eps"],
        bool(params["use_revin"]),
    )


def training_objective(model, batch, criterion):
    forecast, loss, _ = model.training_objective(batch.x, batch.target)
    return forecast, loss


SPEC = ModelSpec(
    name="DistDF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("channel_wise_linear", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)

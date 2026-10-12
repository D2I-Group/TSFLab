"""Model specification for AMRC."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.amrc.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    mask_samples: int = Field(default=4, gt=0)
    lambda_aml: float = Field(default=0.1, ge=0)
    lambda_esp: float = Field(default=0.1, ge=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["d_model"],
        params["mask_samples"],
        params["lambda_aml"],
        params["lambda_esp"],
        bool(params["use_revin"]),
    )


def training_objective(model, batch, criterion):
    forecast, loss, _ = model.training_objective(batch.x, batch.target)
    return forecast, loss


SPEC = ModelSpec(
    name="AMRC",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)

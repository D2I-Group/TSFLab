"""Runtime specification for LatentTSF."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.latenttsf.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    d_ff: int = Field(default=128, gt=0)
    mse_weight: float = Field(default=10.0, ge=0)
    cosine_weight: float = Field(default=15.0, ge=0)
    use_latent_norm: bool = True
    kernel_size: int = Field(default=25, gt=0)
    individual: bool = False
    ae_train_epochs: int = Field(default=100, ge=0)
    ae_lr: float = Field(default=5e-4, gt=0)
    ae_loss: str = Field(default="MAE", pattern="^(MAE|MSE)$")


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        mse_weight=params["mse_weight"],
        cosine_weight=params["cosine_weight"],
        use_latent_norm=params["use_latent_norm"],
        kernel_size=params["kernel_size"],
        individual=params["individual"],
        ae_train_epochs=params["ae_train_epochs"],
        ae_lr=params["ae_lr"],
        ae_loss=params["ae_loss"],
    )


def training_objective(model, batch, criterion):
    forecast, loss, _ = model.training_objective(batch.x, batch.target)
    return forecast, loss


SPEC = ModelSpec(
    name="LatentTSF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_latenttsf.toml",
    capabilities=frozenset(["time-series", "pretraining-stage"]),
    components=("dlinear",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)

"""Runtime specification for TimeAlign."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timealign.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_num: int = Field(default=4, gt=0)
    d_model: int = Field(default=32, gt=0)
    d_ff: int = Field(default=32, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    pos: bool = True
    layer_norm: bool = True
    loc: bool = True
    glo: bool = True
    local_margin: float = Field(default=0.0, ge=0)
    global_margin: float = Field(default=0.0, ge=0)
    w_recon: float = Field(default=1.0, ge=0)
    w_align: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def _one_alignment_mode(self):
        if not (self.loc or self.glo):
            raise ValueError("at least one alignment mode must be enabled")
        return self


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        patch_num=params["patch_num"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        e_layers=params["e_layers"],
        dropout=params["dropout"],
        pos=params["pos"],
        layer_norm=params["layer_norm"],
        loc=params["loc"],
        glo=params["glo"],
        local_margin=params["local_margin"],
        global_margin=params["global_margin"],
        w_recon=params["w_recon"],
        w_align=params["w_align"],
    )


def training_objective(model, batch, criterion):
    forecast, loss, _ = model.training_objective(batch.x, batch.target)
    return forecast, loss


SPEC = ModelSpec(
    name="TimeAlign",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_timealign.toml",
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)

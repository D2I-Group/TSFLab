"""Runtime specification for Fredformer."""
from pydantic import BaseModel, Field, model_validator
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.fredformer.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    band_width: int = Field(default=16, gt=0)
    model_width: int = Field(default=48, gt=0)
    depth: int = Field(default=2, gt=0)
    heads: int = Field(default=6, gt=0)
    feedforward: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    stride: int = Field(default=16, gt=0)
    head_dim: int = Field(default=64, gt=0)
    head_dropout: float = Field(default=0.0, ge=0, lt=1)
    @model_validator(mode="after")
    def architecture(self):
        if self.model_width % self.heads: raise ValueError("model_width must be divisible by heads")
        return self

def build_model(cfg, params): return Model(cfg.task.seq_len, cfg.task.pred_len, **params)
SPEC = ModelSpec(name="Fredformer", module="tsflab.models.fredformer", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/Fredformer.toml", model_card="src/tsflab/models/fredformer/README.md",
    smoke_config=None, capabilities=frozenset(["time-series"]), components=("revin",),
    contract_task={"seq_len":96,"pred_len":96,"label_len":0})

"""Runtime specification for Times2D."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.times2d.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    periods: list[int] = Field(default_factory=lambda: [720, 360, 110, 96, 48], min_length=1)
    patch_lens: list[int] = Field(default_factory=lambda: [48, 32, 16, 6, 3], min_length=1)
    d_model: int = Field(default=64, gt=0)
    n_heads: int = Field(default=16, gt=0)
    e_layers: int = Field(default=3, gt=0)
    d_ff: int = Field(default=64, gt=0)
    num_kernels: int = Field(default=6, gt=0)
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    attn_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    affine: bool = False
    subtract_last: bool = False

    @model_validator(mode="after")
    def _paired_periods(self):
        if len(self.periods) != len(self.patch_lens):
            raise ValueError("periods and patch_lens must have the same length")
        if min(self.periods) < 1 or min(self.patch_lens) < 1:
            raise ValueError("periods and patch_lens must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="Times2D",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_times2d.toml",
    capabilities=frozenset({"time-series"}),
    components=("inception_block", "positional_encoding", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

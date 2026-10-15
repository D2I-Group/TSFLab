"""Runtime specification for SiamTST."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.siamtst.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    The pre-training stage needs ``task.pred_len >= stride`` (the Siamese view is
    shifted into the training horizon) and at least two patches.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_size: int = Field(default=16, gt=0)
    stride: int = Field(default=16, gt=0)
    d_model: int = Field(default=64, gt=0)
    n_heads: int = Field(default=4, gt=0)
    e_layers: int = Field(default=4, gt=0)
    d_ff: int = Field(default=256, gt=0)
    attn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ffn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    pre_norm: bool = True
    qk_norm: bool = True
    bias: bool = False
    min_mask_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    max_mask_ratio: float = Field(default=0.55, ge=0.0, lt=1.0)
    alpha: float = Field(default=0.2, ge=0.0, le=1.0)
    pretrain_epochs: int = Field(default=30, ge=0)
    pretrain_lr: float = Field(default=1e-3, gt=0.0)
    pretrain_weight_decay: float = Field(default=0.1, ge=0.0)
    freeze_backbone: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.min_mask_ratio > self.max_mask_ratio:
            raise ValueError("min_mask_ratio must not exceed max_mask_ratio")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="SiamTST",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_siamtst.toml",
    capabilities=frozenset({"time-series", "pretraining-stage"}),
    components=("revin",),
    contract_task={"seq_len": 512, "pred_len": 96, "label_len": 0},
)

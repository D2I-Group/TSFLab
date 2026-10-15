"""Runtime specification for MoU."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mou.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=16, gt=0)
    d_ff: int = Field(default=256, gt=0)
    e_layers: int = Field(default=1, gt=0)
    num_experts: int = Field(default=4, gt=0)
    top_k: int = Field(default=2, gt=0)
    d_state: int = Field(default=21, gt=0)
    ffn_expand: int = Field(default=2, gt=0)
    dropout: float = 0.1
    mamba_dropout: float = 0.1
    ffn_dropout: float = 0.2
    block_dropout: float = 0.2
    conv_dropout: float = 0.3
    attn_dropout: float = 0.0
    tf_dropout: float = 0.2
    head_dropout: float = 0.0
    use_revin: bool = True

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_ff=params["d_ff"],
        e_layers=params["e_layers"],
        num_experts=params["num_experts"],
        top_k=params["top_k"],
        d_state=params["d_state"],
        ffn_expand=params["ffn_expand"],
        dropout=params["dropout"],
        mamba_dropout=params["mamba_dropout"],
        ffn_dropout=params["ffn_dropout"],
        block_dropout=params["block_dropout"],
        conv_dropout=params["conv_dropout"],
        attn_dropout=params["attn_dropout"],
        tf_dropout=params["tf_dropout"],
        head_dropout=params["head_dropout"],
        use_revin=params["use_revin"],
    )


SPEC = ModelSpec(
    name="MoU",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mou.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "mamba", "positional_encoding", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

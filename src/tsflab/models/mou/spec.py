"""Runtime specification for MoU."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mou.model import Model

"""Validated parameters for MoU."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 16
    d_ff: int = 256
    e_layers: int = 1
    num_experts: int = 4
    top_k: int = 2
    d_state: int = 21
    ffn_expand: int = 2
    dropout: float = 0.1
    mamba_dropout: float = 0.1
    ffn_dropout: float = 0.2
    block_dropout: float = 0.2
    conv_dropout: float = 0.3
    attn_dropout: float = 0.0
    tf_dropout: float = 0.2
    head_dropout: float = 0.0
    use_revin: bool = True


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

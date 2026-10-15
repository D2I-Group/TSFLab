"""Runtime specification for LLM4TS."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelArtifact, ModelSpec
from tsflab.models.llm4ts.model import Model

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

DEFAULTS = dict(
    patch_len=16,
    stride=8,
    first_k_layers=6,
    dropout=0.05,
    token_embed_type="conv",
    temporal_embed_type="learned",
    freq="h",
    lora_r=8,
    lora_alpha=64.0,
    lora_dropout=0.0,
    align_epochs=5,
    align_lr=7.912045141879411e-05,
    align_weight_decay=0.0005542494992024964,
    probe_epochs=7,
    probe_lr=1.8257759510439175e-05,
    probe_weight_decay=0.0014555863788252605,
)


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    first_k_layers: int = Field(default=6, ge=1, le=12)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    token_embed_type: Literal["conv", "linear"] = "conv"
    temporal_embed_type: Literal["learned", "fixed", "none"] = "learned"
    freq: Literal["h", "t"] = "h"
    lora_r: int = Field(default=8, gt=0)
    lora_alpha: float = Field(default=64.0, gt=0.0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    align_epochs: int = Field(default=5, ge=0)
    align_lr: float = Field(default=7.912045141879411e-05, gt=0.0)
    align_weight_decay: float = Field(default=0.0005542494992024964, ge=0.0)
    probe_epochs: int = Field(default=7, ge=0)
    probe_lr: float = Field(default=1.8257759510439175e-05, gt=0.0)
    probe_weight_decay: float = Field(default=0.0014555863788252605, ge=0.0)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        **{key: params.get(key, default) for key, default in DEFAULTS.items()},
    )


def build_pretrained_model(cfg, params, paths):
    """Construct LLM4TS and load the verified local GPT-2 checkpoint."""
    model = build_model(cfg, params)
    model.load_llm_weights(paths["gpt2"])
    return model


SPEC = ModelSpec(
    name="LLM4TS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset({"time-series", "pretraining-stage"}),
    components=("embed", "gpt2_backbone", "marks", "revin"),
    contract_task={"seq_len": 336, "pred_len": 96, "label_len": 0},
    artifacts=(
        ModelArtifact(
            name="gpt2",
            url=f"hf://openai-community/gpt2@{GPT2_REVISION}/model.safetensors",
            revision=GPT2_REVISION,
            sha256="248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
            filename="model.safetensors",
            required=True,
        ),
    ),
    artifact_factory=build_pretrained_model,
)

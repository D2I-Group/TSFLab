"""Runtime specification for TALON."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelArtifact, ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models._components.gpt2_backbone import load_gpt2_weights
from tsflab.models.talon.model import Model

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

DEFAULTS = dict(
    token_len=96,
    top_k=3,
    expert_hidden=1024,
    mlp_hidden_dim=1024,
    mlp_hidden_layers=2,
    mlp_activation="tanh",
    dropout=0.1,
    llm_layers=6,
    moe_weight=0.02,
)


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    token_len: int = Field(default=96, ge=3)
    top_k: int = Field(default=3, ge=1, le=3)
    expert_hidden: int = Field(default=1024, gt=0)
    mlp_hidden_dim: int = Field(default=1024, gt=0)
    mlp_hidden_layers: int = Field(default=2, ge=0)
    mlp_activation: Literal["tanh", "relu", "gelu"] = "tanh"
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    llm_layers: int = Field(default=6, ge=1, le=12)
    moe_weight: float = Field(default=0.02, ge=0.0)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        **{key: params.get(key, default) for key, default in DEFAULTS.items()},
    )


def build_pretrained_model(cfg, params, paths):
    """Construct TALON and load the verified local GPT-2 checkpoint (first ``llm_layers`` blocks)."""
    model = build_model(cfg, params)
    load_gpt2_weights(model.llm, paths["gpt2"])
    return model


def training_objective(model, batch: TrainingBatch, criterion):
    """Next-segment supervision of every context position plus ``alpha * L_MoE`` (Eq. 14)."""
    x, future = batch.x, batch.y[:, -batch.pred_len :]
    if batch.features == "MS":
        x, future = x[..., -1:], future[..., -1:]
    return None, model.training_loss(x, future, criterion)


SPEC = ModelSpec(
    name="TALON",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset({"time-series"}),
    components=("gpt2_backbone", "revin", "segment_mlp"),
    contract_task={"seq_len": 192, "pred_len": 96, "label_len": 0},
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
    training_objective=training_objective,
)

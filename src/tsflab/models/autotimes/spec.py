"""Runtime specification for AutoTimes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelArtifact, ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models._components.gpt2_backbone import load_gpt2_weights
from tsflab.models.autotimes.model import Model

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    token_len: int = Field(default=96, gt=0)
    mlp_hidden_dim: int = Field(default=512, gt=0)
    mlp_hidden_layers: int = Field(default=2, ge=0)
    mlp_activation: Literal["tanh", "relu", "gelu"] = "tanh"
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    llm_layers: int = Field(default=12, ge=1, le=12)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        token_len=params["token_len"],
        mlp_hidden_dim=params["mlp_hidden_dim"],
        mlp_hidden_layers=params["mlp_hidden_layers"],
        mlp_activation=params["mlp_activation"],
        dropout=params["dropout"],
        llm_layers=params["llm_layers"],
    )


def build_pretrained_model(cfg, params, paths):
    """Construct AutoTimes and load the verified local GPT-2 checkpoint."""
    model = build_model(cfg, params)
    load_gpt2_weights(model.llm, paths["gpt2"])
    return model


def training_objective(model, batch: TrainingBatch, criterion):
    """Next-segment supervision of every context position (paper Eq. 8)."""
    x, future = batch.x, batch.y[:, -batch.pred_len :]
    if batch.features == "MS":
        x, future = x[..., -1:], future[..., -1:]
    return None, model.next_segment_loss(x, future, criterion)


SPEC = ModelSpec(
    name="AutoTimes",
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

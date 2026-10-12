"""Runtime specification for CALF."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelArtifact, ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models._components.gpt2_backbone import read_safetensors
from tsflab.models.calf.model import Model

# openai-community/gpt2 (MIT), pinned commit and the LFS SHA-256 of its weights.
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_WEIGHTS = ModelArtifact(
    name="gpt2",
    url=f"hf://openai-community/gpt2@{GPT2_REVISION}/model.safetensors",
    revision=GPT2_REVISION,
    sha256="248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
    filename="model.safetensors",
    required=False,
)


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1, le=1024)
    gpt_layers: int = Field(default=6, ge=1, le=12)
    n_principal: int = Field(default=500, ge=1, le=768)
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: float = 32.0
    lora_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    feature_weight: float = Field(default=0.01, ge=0.0)
    output_weight: float = Field(default=1.0, ge=0.0)
    feature_decay: float = Field(default=0.8, gt=0.0)
    alignment_loss: Literal["l1", "smooth_l1", "mse"] = "l1"


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        gpt_layers=params["gpt_layers"],
        n_principal=params["n_principal"],
        lora_rank=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        dropout=params["dropout"],
        feature_weight=params["feature_weight"],
        output_weight=params["output_weight"],
        feature_decay=params["feature_decay"],
        alignment_loss=params["alignment_loss"],
    )


def build_pretrained(cfg, params, paths):
    """Load the verified local GPT-2 weights when present; else random init."""
    model = build_model(cfg, params)
    if "gpt2" in paths:
        model.load_gpt2(read_safetensors(paths["gpt2"]))
    return model


def training_objective(model, batch: TrainingBatch, criterion):
    """Supervised criterion on Y_time plus feature and output alignment (Eq. 6)."""
    outputs = model.cross_modal_outputs(batch.x)
    forecast = outputs["time"]
    loss = criterion(batch.align(forecast), batch.target)
    return forecast, loss + model.cross_modal_loss(outputs)


SPEC = ModelSpec(
    name="CALF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_calf.toml",
    capabilities=frozenset({"time-series"}),
    components=("gpt2_backbone", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    artifacts=(GPT2_WEIGHTS,),
    artifact_factory=build_pretrained,
    training_objective=training_objective,
)

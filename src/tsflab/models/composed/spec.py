"""Model specification for Composed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models._slots.registry import DEFAULTS, check_assignment
from tsflab.models.composed.model import Model


class ModelParameterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    normalization: Literal["none", "revin", "last_value_center"] = "none"
    decomposition: Literal["none", "series_decomposition"] = "none"
    temporal: Literal[
        "linear",
        "channel_wise_linear",
        "tst_transformer",
        "mamba",
        "gated_dilated_conv",
        "mixer_block",
    ] = "linear"
    channel: Literal["independent", "individual", "mixing"] = "independent"
    head: Literal["flatten_forecast_head"] = "flatten_forecast_head"
    hidden: int = Field(default=16, gt=0)
    n_layers: int = Field(default=1, gt=0)
    n_heads: int = Field(default=4, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    kernel_size: int = Field(default=25, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @field_validator("decomposition", mode="before")
    @classmethod
    def _decomposition_is_executable(cls, value):
        if value == "wavelet":
            raise ValueError(
                "decomposition 'wavelet' is not executable in Composed: the wavelet component has no "
                "uniform [B, L, C] branch interface; use 'none' or 'series_decomposition'"
            )
        return value

    @field_validator("head", mode="before")
    @classmethod
    def _head_is_point_output(cls, value):
        if value in ("quantile_head", "gaussian_parameter_head"):
            raise ValueError(
                f"head {value!r} is not available in Composed (point output only); "
                "register a dedicated model with `tsf model compose <spec> --register NAME`"
            )
        return value

    @model_validator(mode="after")
    def _slots_are_compatible(self):
        problems = check_assignment(
            {
                "normalization": self.normalization,
                "decomposition": self.decomposition,
                "temporal": self.temporal,
                "channel": self.channel,
                "head": self.head,
                "loss": DEFAULTS["loss"],
            }
        )
        problems = [p for p in problems if "needs loss" not in p]  # the loss lives in training.loss
        if self.kernel_size % 2 == 0:
            problems.append("kernel_size must be odd")
        if self.hidden % self.n_heads:
            problems.append("hidden must be divisible by n_heads")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def build_model(cfg, params):
    """Construct Composed from a validated run configuration."""
    options = {k: v for k, v in params.items() if k != "enc_in"}
    return Model(
        c_in=params["enc_in"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        **options,
    )


SPEC = ModelSpec(
    name="Composed",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=(
        "channel_wise_linear",
        "flatten_forecast_head",
        "gated_dilated_conv",
        "last_value_center",
        "mamba",
        "mixer_block",
        "positional_encoding",
        "revin",
        "series_decomposition",
        "tst_transformer",
    ),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

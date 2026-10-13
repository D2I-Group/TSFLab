"""Evaluation metrics, quantile levels, and rolling-forecast schema."""

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RollingConfig(BaseModel):
    """Optional rolling-forecast sub-config.

    Only consulted when ``EvaluationConfig.strategy == "rolling"``. All fields
    are optional; the defaults keep the rolling walk as permissive as possible
    (predict ``pred_len``, advance by 1 step, continue until the test data is
    exhausted).
    """

    model_config = ConfigDict(extra="forbid")

    horizon: Optional[int] = None
    stride: int = 1
    num_rollings: Optional[int] = None


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: Literal["val", "test"] = "test"

    metrics: list[str] = Field(
        default_factory=lambda: ["mae", "mse", "rmse", "mape", "mspe"]
    )
    # Canonical quantile levels (the single source of truth for probabilistic
    # forecasting). Threaded into the evaluator and, for the "quantile" loss
    # only, injected into ``training.loss_params`` by ``run_one``. Ignored by
    # point models, so existing runs are unaffected.
    quantile_levels: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    @field_validator("quantile_levels")
    @classmethod
    def _validate_quantile_levels(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("evaluation.quantile_levels must not be empty")
        for level in v:
            if math.isnan(level) or not (0.0 < level < 1.0):
                raise ValueError(
                    "evaluation.quantile_levels must all lie in the open "
                    f"interval (0, 1); got {level!r}"
                )
        if any(a >= b for a, b in zip(v, v[1:])):
            raise ValueError(
                "evaluation.quantile_levels must be strictly increasing; "
                f"got {v!r}"
            )
        return v

    # A frozen checkpoint evaluates without pretraining or optimizer steps.
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    enable_profile: bool = False
    # Evaluation strategy. "fixed" (default) keeps the historical fixed-window
    # evaluation untouched. "rolling" opts into a TFB-style rolling forecast.
    strategy: Literal["fixed", "rolling"] = "fixed"
    rolling: RollingConfig = Field(default_factory=RollingConfig)

"""Forecasting task dimensions and declarative data-setting schema."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskConfig(BaseModel):
    """Task configuration.

    All modes perform forecasting; ``mode`` selects the *data setting* (how a
    batch is shaped and what the model receives).

    ``mode`` is executable: config loading rejects datasets or models whose
    registered runtime contracts do not support the selected data setting.

    Parameters
    ----------
    mode : str
        Forecasting data setting, one of:

        * ``"time_series"`` (default) — classic multivariate time-series
          forecasting. Batches are ``(B, T, C)`` value tensors; every channel
          is a target.
        * ``"spatiotemporal"`` — node-structured forecasting. Batches carry a
          ``(B, T, N, 1 + F)`` tensor where channel 0 is the value and the
          remaining ``F`` channels are per-node covariates / calendar features.
          Only the value channel of all ``N`` nodes is the target.
        * ``"covariate"`` — like ``"spatiotemporal"`` but the model also
          receives the *future* (known) covariate block ``(B, pred_len, N, F)``
          on the decoder side. Used e.g. by the air-quality forecasters
          CauAir / AirCade, which consume future weather covariates.

    seq_len, label_len, pred_len : int
        Window lengths.
    features : str
        Feature mode ("M", "S", "MS"); only used by ``time_series``.
    inverse : bool
        Whether to inverse-transform predictions before metrics.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["time_series", "spatiotemporal", "covariate"] = "time_series"
    seq_len: int = Field(gt=0)
    label_len: int = Field(ge=0)
    pred_len: int = Field(gt=0)
    features: str = "M"
    inverse: bool = False

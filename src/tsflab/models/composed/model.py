"""Slot-assignment forecaster; the adapters live in ``tsflab.models._slots``."""

from __future__ import annotations

# Each cataloged component the adapters can instantiate is imported here so the
# component audit sees the declared set (the adapters import them as well).
from tsflab.models._components.channel_wise_linear import ChannelWiseLinear  # noqa: F401
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead  # noqa: F401
from tsflab.models._components.gated_dilated_conv import gated_dilated_conv  # noqa: F401
from tsflab.models._components.last_value_center import center_on_last_value  # noqa: F401
from tsflab.models._components.mamba import MambaResidualBlock  # noqa: F401
from tsflab.models._components.mixer_block import MixerBlock  # noqa: F401
from tsflab.models._components.positional_encoding import positional_encoding  # noqa: F401
from tsflab.models._components.revin import RevIN  # noqa: F401
from tsflab.models._components.series_decomposition import SeriesDecomposition  # noqa: F401
from tsflab.models._components.tst_transformer import TSTEncoder  # noqa: F401
from tsflab.models._slots.adapters import Pipeline
from tsflab.models._slots.registry import canonical


class Model(Pipeline):
    """Point forecaster whose six slots are chosen by parameters."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        *,
        normalization: str = "none",
        decomposition: str = "none",
        temporal: str = "linear",
        channel: str = "independent",
        head: str = "flatten_forecast_head",
        features: str = "M",
        **options,
    ) -> None:
        assignment = {
            "normalization": canonical("normalization", normalization),
            "decomposition": canonical("decomposition", decomposition),
            "temporal": canonical("temporal", temporal),
            "channel": canonical("channel", channel),
            "head": canonical("head", head),
        }
        super().__init__(assignment, c_in, seq_len, pred_len, **options)
        self.features = features

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        y = super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return y[..., -1:] if self.features == "MS" else y

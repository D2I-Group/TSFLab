"""Independent channel-wise patch Transformer forecasting backbone."""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN
from tsflab.models._components.tst_transformer import TSTEncoder


class PatchTSTBackbone(nn.Module):
    """Patch each channel independently, encode patches, and forecast directly."""

    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        padding_patch: str | None,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int | None,
        d_v: int | None,
        d_ff: int,
        activation: str,
        norm: str,
        attn_dropout: float,
        res_dropout: float,
        ffn_dropout: float,
        proj_dropout: float,
        head_dropout: float,
        pre_norm: bool,
        pe: str,
        learn_pe: bool,
        head_type: str,
        individual: bool,
        revin: bool,
        affine: bool,
        subtract_last: bool,
    ) -> None:
        super().__init__()
        if patch_len < 1 or stride < 1 or patch_len > context_window:
            raise ValueError("patch_len and stride must define at least one history patch")
        if padding_patch not in {None, "end"}:
            raise ValueError("padding_patch must be None or 'end'")
        if head_type != "flatten":
            raise ValueError("only the canonical flatten forecast head is supported")
        self.context_window = context_window
        self.patch_len = patch_len
        self.stride = stride
        self.pad_end = padding_patch == "end"
        self.normalizer = RevIN(c_in, affine=affine, subtract_last=subtract_last, enabled=revin)
        effective_length = context_window + (stride if self.pad_end else 0)
        patch_count = 1 + (effective_length - patch_len) // stride
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = positional_encoding(pe, learn_pe, patch_count, d_model)
        self.input_dropout = nn.Dropout(res_dropout)
        self.encoder = TSTEncoder(
            d_model,
            n_heads,
            n_layers=n_layers,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            activation=activation,
            norm=norm,
            attn_dropout=attn_dropout,
            res_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
            proj_dropout=proj_dropout,
            pre_norm=pre_norm,
        )
        self.head = FlattenForecastHead(
            individual,
            c_in,
            d_model * patch_count,
            target_window,
            head_dropout=head_dropout,
        )

    def forward(self, values: torch.Tensor, *_: object) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.context_window:
            raise ValueError(f"expected [batch, {self.context_window}, channels], got {tuple(values.shape)}")
        normalized = self.normalizer(values, "norm")
        channels_first = normalized.transpose(1, 2)
        if self.pad_end:
            channels_first = torch.nn.functional.pad(channels_first, (0, self.stride), mode="replicate")
        patches = channels_first.unfold(-1, self.patch_len, self.stride)
        embedded = self.patch_projection(patches)
        batch, channels, patch_count, width = embedded.shape
        tokens = embedded.reshape(batch * channels, patch_count, width)
        encoded = self.encoder(self.input_dropout(tokens + self.position))
        features = encoded.reshape(batch, channels, patch_count, width).permute(0, 1, 3, 2)
        forecast = self.head(features).transpose(1, 2)
        return self.normalizer(forecast, "denorm")

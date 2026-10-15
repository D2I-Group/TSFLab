"""PatchTSMixer / TSMixer (Ekambaram et al., KDD 2023, arXiv:2306.09364).

Independent implementation of the supervised forecasting path: RevIN, patching,
linear patch embedding, stacked MLP-Mixer layers (inter-patch, intra-patch
feature, and optional inter-channel mixing, each with a gated-attention block),
and a flatten-linear prediction head shared across channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.revin import RevIN
from tsflab.models._components.softmax_gate import SoftmaxGate
from tsflab.models._inputs import check_window


class MixerMLP(nn.Module):
    """Linear(dim -> dim * expansion), GELU, Dropout, Linear back, Dropout (Fig. 6a)."""

    def __init__(self, dim: int, expansion_factor: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion_factor)
        self.fc2 = nn.Linear(dim * expansion_factor, dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(nn.functional.gelu(self.fc1(x)))))


class AxisMixer(nn.Module):
    """Pre-LayerNorm residual MLP-Mixer sub-block over one axis of [B, C, P, D].

    LayerNorm acts on the hidden feature axis ``D`` for every axis choice. The
    chosen axis is moved last, mixed by an MLP then a gated-attention block
    (paper Sec. 3.3.5: the gate follows the MLP), moved back, and added to the
    block input.
    """

    def __init__(
        self, axis: str, dim: int, d_model: int, expansion_factor: int, dropout: float, gated_attn: bool
    ) -> None:
        super().__init__()
        if axis not in ("patch", "feature", "channel"):
            raise ValueError("axis must be 'patch', 'feature', or 'channel'")
        self.axis = axis
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = MixerMLP(dim, expansion_factor, dropout)
        self.gate = SoftmaxGate(dim) if gated_attn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, P, D]
        h = self.norm(x)
        if self.axis == "patch":
            h = h.transpose(2, 3)  # [B, C, D, P]
        elif self.axis == "channel":
            h = h.permute(0, 3, 2, 1)  # [B, D, P, C]
        h = self.mlp(h)
        if self.gate is not None:
            h = self.gate(h)
        if self.axis == "patch":
            h = h.transpose(2, 3)
        elif self.axis == "channel":
            h = h.permute(0, 3, 2, 1)
        return x + h


class MixerLayer(nn.Module):
    """One mixer layer: [channel mixer in mix_channel mode], patch mixer, feature mixer."""

    def __init__(
        self,
        mode: str,
        enc_in: int,
        num_patches: int,
        d_model: int,
        expansion_factor: int,
        dropout: float,
        gated_attn: bool,
    ) -> None:
        super().__init__()
        self.channel_mixer = (
            AxisMixer("channel", enc_in, d_model, expansion_factor, dropout, gated_attn)
            if mode == "mix_channel"
            else None
        )
        self.patch_mixer = AxisMixer("patch", num_patches, d_model, expansion_factor, dropout, gated_attn)
        self.feature_mixer = AxisMixer("feature", d_model, d_model, expansion_factor, dropout, gated_attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_mixer is not None:
            x = self.channel_mixer(x)
        return self.feature_mixer(self.patch_mixer(x))


class Model(nn.Module):
    """Supervised PatchTSMixer forecaster: [B, L, C] history -> [B, H, C] forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 16,
        patch_len: int = 16,
        stride: int = 8,
        num_layers: int = 3,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        mode: str = "common_channel",
        gated_attn: bool = True,
    ) -> None:
        super().__init__()
        if seq_len <= patch_len:
            raise ValueError("seq_len must exceed patch_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride, self.mode = patch_len, stride, mode
        self.num_patches = (seq_len - patch_len) // stride + 1
        # Patching keeps the most recent samples; the oldest remainder is dropped.
        self.start = seq_len - (patch_len + stride * (self.num_patches - 1))

        self.revin = RevIN(enc_in, affine=False)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.layers = nn.ModuleList(
            MixerLayer(mode, enc_in, self.num_patches, d_model, expansion_factor, dropout, gated_attn)
            for _ in range(num_layers)
        )
        self.head_drop = nn.Dropout(head_dropout)
        self.head = FlattenForecastHead(False, enc_in, self.num_patches * d_model, pred_len, 0.0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, C] -> [B, C, P, patch_len] (paper Sec. 3.3.2)."""
        return x[:, self.start :, :].unfold(1, self.patch_len, self.stride).permute(0, 2, 1, 3)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.enc_in)
        hidden = self.patch_embedding(self.patchify(self.revin(x_enc, "norm")))
        for layer in self.layers:
            hidden = layer(hidden)
        forecast = self.head(self.head_drop(hidden))  # [B, C, H]
        return self.revin(forecast.transpose(1, 2), "denorm")

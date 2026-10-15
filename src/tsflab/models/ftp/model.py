"""Clean-room FusionTimePatch (FTP) from the AAAI 2026 paper.

The implementation preserves pure-MLP Dual-GLF channel-independent/channel-
mixed recursion, deterministic Channel Enhancement, tri-stream fusion, and the
final prediction head. It does not inspect or depend on the reference source.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def _right_padded_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """Paper patch count floor((L-P)/S)+2, including one padded tail patch."""
    remainder = (x.shape[-1] - patch_len) % stride
    padding = stride - remainder
    return F.pad(x, (0, padding), mode="replicate").unfold(-1, patch_len, stride)


class _GlobalLocalLevel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        d_model: int,
        patch_len: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.ci_local = nn.Linear(patch_len, d_model)
        self.ci_global = nn.Linear(seq_len, d_model)
        self.ci_output = nn.Linear(2 * d_model, seq_len)
        self.cm_local = nn.Linear(enc_in * patch_len, enc_in * d_model)
        self.cm_global = nn.Linear(enc_in * seq_len, enc_in * d_model)
        self.cm_output = nn.Linear(2 * d_model, seq_len)

    def ci(self, x: torch.Tensor) -> torch.Tensor:
        patches = _right_padded_patches(x, self.patch_len, self.stride)
        local = self.ci_local(patches).mean(-2)
        global_feature = self.ci_global(x)
        return x + self.ci_output(torch.cat((local, global_feature), dim=-1))

    def cm(self, x: torch.Tensor) -> torch.Tensor:
        patches = _right_padded_patches(x, self.patch_len, self.stride)
        batch, channels, count, patch = patches.shape
        local = patches.permute(0, 2, 1, 3).reshape(batch, count, channels * patch)
        local = self.cm_local(local).reshape(batch, count, channels, -1).mean(1)
        global_feature = self.cm_global(x.flatten(1)).reshape(batch, channels, -1)
        return x + self.cm_output(torch.cat((local, global_feature), dim=-1))


class _FTPEncoderLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        d_model: int,
        patch_unit: int,
        num_scales: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        patch_lengths = [min(seq_len, patch_unit * (index + 1)) for index in range(num_scales)]
        self.levels = nn.ModuleList(
            _GlobalLocalLevel(seq_len, enc_in, d_model, patch_len, min(stride, patch_len))
            for patch_len in patch_lengths
        )
        self.ce_embedding = nn.Linear(seq_len, d_model)
        self.ce_latent_score = nn.Linear(d_model, d_model)
        self.ce_channel_score = nn.Linear(d_model, 1, bias=False)
        self.ce_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.ci_projection = nn.Linear(seq_len, d_model)
        self.cm_projection = nn.Linear(seq_len, d_model)
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.original_embedding = nn.Linear(seq_len, d_model)
        self.sequence_projection = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, seq_len),
        )

    def channel_enhancement(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.ce_embedding(x)
        latent_weights = self.ce_latent_score(embedded).softmax(-1)
        salient = embedded * (1 + latent_weights * embedded.shape[-1])
        channel_weights = self.ce_channel_score(salient).squeeze(-1).softmax(-1)
        dominant = torch.einsum("bc,bcd->bd", channel_weights, salient)
        enhanced = salient + self.ce_mlp(dominant).unsqueeze(1)
        return enhanced, channel_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ci, cm = x, x
        for level in self.levels:
            ci = level.ci(ci)
            cm = level.cm(cm)
        enhanced, _ = self.channel_enhancement(x)
        fused = self.fusion(torch.cat((self.ci_projection(ci), self.cm_projection(cm), enhanced), dim=-1))
        return self.sequence_projection(torch.cat((fused, self.original_embedding(x)), dim=-1))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        num_layers: int = 2,
        patch_unit: int = 4,
        num_scales: int = 3,
        stride: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_revin = use_revin
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.layers = nn.ModuleList(
            _FTPEncoderLayer(
                seq_len,
                enc_in,
                d_model,
                min(patch_unit, seq_len),
                num_scales,
                min(stride, seq_len),
                dropout,
            )
            for _ in range(num_layers)
        )
        self.head_embedding = nn.Linear(seq_len, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, pred_len),
        )

    def _validate(self, x: torch.Tensor) -> None:
        check_window(x, self.seq_len, self.enc_in)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        self._validate(x_enc)
        normalized = self.revin(x_enc, "norm") if self.use_revin else x_enc
        representation = normalized.transpose(1, 2)
        for layer in self.layers:
            representation = layer(representation)
        forecast = self.head(self.head_embedding(representation)).transpose(1, 2)
        return self.revin(forecast, "denorm") if self.use_revin else forecast

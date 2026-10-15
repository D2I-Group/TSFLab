"""PatchMLP model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._inputs import check_window
from tsflab.models.patchmlp.layers import Emb, Encoder, SeriesDecomp


def _check_patches(seq_len: int, pred_len: int, patch_len: list[int]) -> None:
    if seq_len < 2:
        raise ValueError("seq_len >= 2 is required")
    if any(patch > seq_len for patch in patch_len):
        raise ValueError("each patch length must be in [2, seq_len]")


def _check_patch_counts(seq_len: int, d_model: int, patch_len: list[int]) -> None:
    scale_width = d_model // 4
    patch_counts = [int((seq_len - patch) / (patch // 2) + 1) for patch in patch_len]
    if any(scale_width < count for count in patch_counts):
        raise ValueError("d_model / 4 must be at least the patch count of every scale")


class PatchMLPModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        use_norm: bool = True,
        moving_avg: int = 13,
        patch_len: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_norm = use_norm
        self.decomposition = SeriesDecomp(moving_avg)
        patch_len = patch_len or [48, 24, 12, 6]
        _check_patches(seq_len, pred_len, patch_len)
        _check_patch_counts(seq_len, d_model, patch_len)
        self.emb = Emb(seq_len, d_model, patch_len)
        self.residual_layers = nn.ModuleList([Encoder(d_model, enc_in, channel_mixing=False) for _ in range(e_layers)])
        self.smooth_layers = nn.ModuleList([Encoder(d_model, enc_in, channel_mixing=True) for _ in range(e_layers)])
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.enc_in)
        if self.use_norm:
            x_enc, means, stdev = standardize(x_enc)

        x = x_enc.permute(0, 2, 1)
        x = self.emb(x)
        residual, smooth = self.decomposition(x)

        for mod in self.residual_layers:
            residual = mod(residual)
        for mod in self.smooth_layers:
            smooth = mod(smooth)

        x = residual + smooth
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = destandardize(dec_out, means, stdev)

        return dec_out

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.forecast(x_enc)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        use_norm: bool,
        moving_avg: int,
        patch_len: list[int] | None,
    ) -> None:
        super().__init__()
        self.model = PatchMLPModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            e_layers=e_layers,
            use_norm=use_norm,
            moving_avg=moving_avg,
            patch_len=patch_len,
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

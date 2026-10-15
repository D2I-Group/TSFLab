"""Quantile PatchTST: wraps the point PatchTST backbone with a monotone quantile head."""

from __future__ import annotations

import torch.nn as nn

from tsflab.models._components.patchtst import PatchTSTBackbone
from tsflab.models._components.quantile_head import QuantileHead, validate_quantile_levels


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str | None = "end",
        e_layers: int = 3,
        d_model: int = 512,
        n_heads: int = 8,
        d_k: int | None = None,
        d_v: int | None = None,
        d_ff: int = 2048,
        activation: str = "gelu",
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        head_dropout: float = 0.0,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = False,
        head_type: str = "flatten",
        individual: bool = False,
        revin: bool = True,
        affine: bool = False,
        subtract_last: bool = False,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.output_type = "quantile"
        levels = validate_quantile_levels(quantile_levels)
        self.backbone = PatchTSTBackbone(
            c_in=enc_in,
            context_window=seq_len,
            target_window=pred_len,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            n_layers=e_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            activation=activation,
            norm=norm,
            attn_dropout=attn_dropout,
            res_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
            proj_dropout=proj_dropout,
            head_dropout=head_dropout,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
        )
        self.quantile_head = QuantileHead(levels, in_features=1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"expected [batch, {self.seq_len}, channels], got {tuple(x_enc.shape)}")
        base = self.backbone(x_enc)  # (B, pred_len, enc_in)
        if self.features == "MS":
            base = base[:, :, -1:]
        base = base[:, -self.pred_len :, :]  # (B, pred_len, c_out)
        return self.quantile_head(base.unsqueeze(-1))  # (B, pred_len, c_out, Q)

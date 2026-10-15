"""Paper-driven local implementation of Sensorformer.

No official implementation was available to inspect: the repository linked
by the paper (https://github.com/BigYellowTiger/Sensorformer) is an empty
GitHub repository (confirmed via the GitHub API on the pinned scaffold date;
no commits, tree, or files exist to fetch). The architecture below is
implemented directly from the paper's Section 3 (Fig. 2, Algorithms 1-2):

1. Patch each variable independently the same way PatchTST does (reused
   `embed.PatchEmbedding`): replicate-pad by `stride`, extract overlapping
   patches, and linearly embed each patch with an added positional encoding.
2. Stack `layers` Sensor Attention Blocks (`global_patch_compression_attention`,
   shared component). Each block treats every variable's last patch as a
   query over the concatenation of all variables' patches (stage 1) to build
   one compressed "Sensor" summary token per variable, then lets every patch
   of every variable attend back to those summaries (stage 2) — extracting
   cross-variable and cross-time dependencies without full quadratic
   cross-patch self-attention.
3. Project the flattened final-layer patch representations of each variable
   to the forecast horizon with a shared linear head (reused
   `flatten_forecast_head.FlattenForecastHead`).
"""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.embed import PatchEmbedding
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.global_patch_compression_attention import (
    GlobalPatchCompressionAttention,
)
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Patch + embed -> stacked Sensor Attention Blocks -> flatten forecast head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        layers: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_len > seq_len + stride:
            raise ValueError("patch_len must not exceed seq_len + stride after padding")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, stride, dropout)
        num_patches = (seq_len - patch_len) // stride + 2
        self.blocks = nn.ModuleList(
            [GlobalPatchCompressionAttention(d_model, n_heads, d_ff, dropout) for _ in range(layers)]
        )
        self.head = FlattenForecastHead(
            individual=False,
            n_vars=enc_in,
            nf=num_patches * d_model,
            target_window=pred_len,
            head_dropout=dropout,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        values = x_enc.permute(0, 2, 1)
        patches, n_vars = self.patch_embedding(values)
        patches = patches.reshape(batch, n_vars, patches.shape[-2], patches.shape[-1])

        for block in self.blocks:
            patches = block(patches)

        return self.head(patches).permute(0, 2, 1)

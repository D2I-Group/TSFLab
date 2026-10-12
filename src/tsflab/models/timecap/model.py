"""Independent TimeCAP forecasting implementation from the AAAI paper.

The implementation retains flexible overlapping channel groups, temporal-first
channel-aware attention, meta-router communication, scatter aggregation, and
the dynamic autoregressive/one-shot dual-head inference path. No code from the
reference repository was inspected or copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        patch_len: int = 16,
        group_size: int = 4,
        group_stride: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_alpha: float = 0.1,
        fusion_midpoint: float | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.group_size = min(group_size, enc_in)
        self.num_patches = math.ceil(seq_len / patch_len)
        self.fusion_alpha = fusion_alpha
        self.fusion_midpoint = (
            float(fusion_midpoint) if fusion_midpoint is not None else (pred_len + 1) / 2
        )

        starts = list(range(0, enc_in, group_stride))
        groups = [
            [(start + offset) % enc_in for offset in range(self.group_size)]
            for start in starts
        ]
        self.register_buffer("group_indices", torch.tensor(groups, dtype=torch.long))
        self.revin = RevIN(enc_in)
        self.group_projections = nn.ModuleList(
            nn.Linear(patch_len, d_model) for _ in groups
        )
        self.meta_routers = nn.Parameter(
            torch.empty(len(groups), self.num_patches, d_model)
        )
        nn.init.normal_(self.meta_routers, std=0.02)
        self.intra_group = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.intra_norm = nn.LayerNorm(d_model)
        self.inter_group = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.inter_norm = nn.LayerNorm(d_model)
        self.one_shot_head = nn.Linear(self.num_patches * d_model, pred_len)
        self.autoregressive_cell = nn.GRUCell(1, d_model)
        self.autoregressive_head = nn.Linear(d_model, 1)

    @staticmethod
    def channel_aware_mask(
        query_patches: torch.Tensor, key_patches: torch.Tensor
    ) -> torch.Tensor:
        """Mask interactions that are not time-aligned (paper Eq. 4/8)."""
        return query_patches[:, None] != key_patches[None, :]

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        values = x.transpose(1, 2)
        padded = self.num_patches * self.patch_len
        if padded > self.seq_len:
            values = F.pad(values, (0, padded - self.seq_len), mode="replicate")
        return values.unfold(-1, self.patch_len, self.patch_len)

    def groupwise_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Return scatter-averaged ``O`` from paper Equation (9)."""
        patches = self._patch(x)
        batch = x.shape[0]
        group_outputs: list[torch.Tensor] = []
        routers: list[torch.Tensor] = []

        intra_patch_ids = torch.arange(self.num_patches, device=x.device).repeat_interleave(
            self.group_size + 1
        )
        intra_mask = self.channel_aware_mask(intra_patch_ids, intra_patch_ids)
        for group_number, indices in enumerate(self.group_indices):
            group = patches[:, indices]
            embedded = self.group_projections[group_number](group)
            temporal_first = embedded.permute(0, 2, 1, 3)
            router = self.meta_routers[group_number].view(
                1, self.num_patches, 1, -1
            ).expand(batch, -1, -1, -1)
            sequence = torch.cat([temporal_first, router], dim=2).flatten(1, 2)
            attended, _ = self.intra_group(
                sequence, sequence, sequence, attn_mask=intra_mask, need_weights=False
            )
            attended = self.intra_norm(sequence + attended).reshape(
                batch, self.num_patches, self.group_size + 1, -1
            )
            group_outputs.append(attended[:, :, : self.group_size])
            routers.append(attended[:, :, -1])

        router_bank = torch.stack(routers, dim=1)
        router_sequence = router_bank.permute(0, 2, 1, 3).flatten(1, 2)
        router_patch_ids = torch.arange(self.num_patches, device=x.device).repeat_interleave(
            len(group_outputs)
        )
        refined_groups: list[torch.Tensor] = []
        for tokens in group_outputs:
            query = tokens.flatten(1, 2)
            query_ids = torch.arange(self.num_patches, device=x.device).repeat_interleave(
                self.group_size
            )
            cross_mask = self.channel_aware_mask(query_ids, router_patch_ids)
            routed, _ = self.inter_group(
                query,
                router_sequence,
                router_sequence,
                attn_mask=cross_mask,
                need_weights=False,
            )
            refined_groups.append(
                self.inter_norm(query + routed).reshape(
                    batch, self.num_patches, self.group_size, -1
                )
            )

        output = x.new_zeros(batch, self.enc_in, self.num_patches, self.meta_routers.shape[-1])
        counts = x.new_zeros(self.enc_in)
        for indices, tokens in zip(self.group_indices, refined_groups, strict=True):
            values = tokens.permute(0, 2, 1, 3)
            for local, channel in enumerate(indices.tolist()):
                output[:, channel] = output[:, channel] + values[:, local]
                counts[channel] = counts[channel] + 1
        return output / counts.view(1, -1, 1, 1).clamp_min(1)

    def dual_head_forecast(
        self, representation: torch.Tensor, normalized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = normalized.shape[0]
        one_shot = self.one_shot_head(representation.flatten(-2)).transpose(1, 2)
        hidden = representation.mean(dim=2).reshape(batch * self.enc_in, -1)
        previous = normalized[:, -1].reshape(batch * self.enc_in, 1)
        steps: list[torch.Tensor] = []
        for _ in range(self.pred_len):
            hidden = self.autoregressive_cell(previous, hidden)
            previous = self.autoregressive_head(hidden)
            steps.append(previous.reshape(batch, self.enc_in))
        autoregressive = torch.stack(steps, dim=1)
        positions = torch.arange(
            1, self.pred_len + 1, device=normalized.device, dtype=normalized.dtype
        )
        weight = torch.sigmoid(
            self.fusion_alpha * (positions - self.fusion_midpoint)
        ).view(1, self.pred_len, 1)
        fused = (1 - weight) * autoregressive + weight * one_shot
        return fused, autoregressive, one_shot

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        representation = self.groupwise_representation(normalized)
        fused, _, _ = self.dual_head_forecast(representation, normalized)
        return self.revin(fused, "denorm")


__all__ = ["Model"]

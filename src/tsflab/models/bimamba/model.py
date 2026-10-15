"""Independent Bi-Mamba+ implementation mapped to Algorithms 1--3 of the paper."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.mamba import MambaBlock
from tsflab.models._inputs import check_window


def patchify(values: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """End-pad and return ``[B, C, patches, patch_len]`` windows."""
    length = values.shape[1]
    count = math.ceil(max(length - patch_len, 0) / stride) + 1
    needed = (count - 1) * stride + patch_len
    if needed > length:
        values = F.pad(values.transpose(1, 2), (0, needed - length), mode="replicate").transpose(1, 2)
    return values.transpose(1, 2).unfold(-1, patch_len, stride)


class SeriesRelationDecider(nn.Module):
    """Differentiable SRA ratio based on positive rank-correlation evidence."""

    def __init__(self, threshold: float = 0.5, temperature: float = 8.0):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] == 1:
            return values.new_zeros(values.shape[0], 1, 1)
        # Soft ranks avoid copying a particular sorting implementation and keep both
        # tokenization paths trainable around the paper's dataset-level decision.
        differences = values.unsqueeze(2) - values.unsqueeze(1)
        ranks = torch.sigmoid(differences).sum(dim=2)
        ranks = ranks - ranks.mean(dim=1, keepdim=True)
        ranks = ranks / ranks.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
        correlation = torch.einsum("blc,bld->bcd", ranks, ranks) / values.shape[1]
        channels = values.shape[-1]
        off_diagonal = ~torch.eye(channels, dtype=torch.bool, device=values.device)
        relation = correlation[:, off_diagonal].clamp_min(0).mean(dim=1)
        return torch.sigmoid((relation - self.threshold) * self.temperature).view(-1, 1, 1)


class MambaPlus(nn.Module):
    """Algorithm 2: selective scan plus complementary forget/new-feature gate."""

    def __init__(self, d_model, d_state, expand, d_conv):
        super().__init__()
        self.scan = MambaBlock(d_model, d_model * expand, math.ceil(d_model / 16), d_conv, d_state)
        self.new_gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        scanned = self.scan(tokens)
        new = torch.sigmoid(self.new_gate(tokens))
        forget = 1.0 - new
        return self.norm(forget * tokens + new * scanned)


class BiMambaPlusEncoder(nn.Module):
    """Algorithm 3: forward/backward Mamba+ fusion and residual FFN."""

    def __init__(self, d_model, d_state, d_ff, expand, d_conv, dropout):
        super().__init__()
        self.forward_block = MambaPlus(d_model, d_state, expand, d_conv)
        self.backward_block = MambaPlus(d_model, d_state, expand, d_conv)
        self.direction_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        forward = self.forward_block(tokens)
        backward = self.backward_block(tokens.flip(1)).flip(1)
        fused = self.direction_norm(tokens + self.dropout(forward + backward))
        return self.output_norm(fused + self.dropout(self.ffn(fused)))


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out=None,
        features="M",
        d_model=128,
        d_state=16,
        e_layers=2,
        expand=2,
        d_conv=4,
        dropout=0.1,
        patch_len=16,
        stride=8,
        d_ff=None,
        sra_threshold=0.5,
    ):
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        self.patch_count = math.ceil(max(seq_len - patch_len, 0) / stride) + 1
        d_ff = d_model * 4 if d_ff is None else d_ff
        self.decider = SeriesRelationDecider(sra_threshold)
        self.independent_projection = nn.Linear(patch_len, d_model)
        self.mixing_projection = nn.Linear(patch_len * enc_in, d_model)
        self.independent_encoder = nn.ModuleList(
            [BiMambaPlusEncoder(d_model, d_state, d_ff, expand, d_conv, dropout) for _ in range(e_layers)]
        )
        self.mixing_encoder = nn.ModuleList(
            [BiMambaPlusEncoder(d_model, d_state, d_ff, expand, d_conv, dropout) for _ in range(e_layers)]
        )
        self.independent_head = nn.Linear(self.patch_count * d_model, pred_len)
        self.mixing_head = nn.Linear(self.patch_count * d_model, pred_len * enc_in)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        values, mean, scale = standardize(x_enc, 1e-5, detach_mean=True, detach_std=True)
        patches = patchify(values, self.patch_len, self.stride)
        batch, channels, count, width = patches.shape

        independent = self.dropout(self.independent_projection(patches)).reshape(batch * channels, count, -1)
        for layer in self.independent_encoder:
            independent = layer(independent)
        independent = (
            self.independent_head(independent.flatten(1)).reshape(batch, channels, self.pred_len).transpose(1, 2)
        )

        mixing_patches = patches.permute(0, 2, 1, 3).reshape(batch, count, channels * width)
        mixing = self.dropout(self.mixing_projection(mixing_patches))
        for layer in self.mixing_encoder:
            mixing = layer(mixing)
        mixing = self.mixing_head(mixing.flatten(1)).reshape(batch, self.pred_len, channels)

        gate = self.decider(values)
        forecast = (1.0 - gate) * independent + gate * mixing
        return destandardize(forecast, mean, scale)

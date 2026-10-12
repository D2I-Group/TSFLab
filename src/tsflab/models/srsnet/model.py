"""Independent SRSNet implementation from the selective patch-space paper."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.revin import RevIN


class SelectivePatching(nn.Module):
    """Score and softly gate contextual patches to form a selective space."""
    def __init__(self, patch_len: int, d_model: int, hidden_size: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.embedding = nn.Linear(patch_len, d_model)
        # A shared final bias would cancel in both centered gates and pairwise
        # ranks, so it is deliberately omitted rather than keeping a dead parameter.
        self.scorer = nn.Sequential(nn.Linear(d_model, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(patches)
        scores = self.scorer(embedded).squeeze(-1)
        gate = torch.sigmoid(self.alpha * (scores - scores.mean(dim=-1, keepdim=True)))
        return embedded * gate.unsqueeze(-1), scores


class DynamicReassembly(nn.Module):
    """Differentiably reorder patches according to their learned utility."""
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def assignment(self, scores: torch.Tensor) -> torch.Tensor:
        count = scores.size(-1)
        pairwise = scores.unsqueeze(-1) - scores.unsqueeze(-2)
        soft_rank = torch.sigmoid(self.alpha * pairwise).sum(dim=-1)
        positions = torch.arange(count, device=scores.device, dtype=scores.dtype)
        logits = -self.alpha * (soft_rank.unsqueeze(-1) - positions).square()
        return logits.softmax(dim=-2)

    def forward(self, patches: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcpi,bcpd->bcid", self.assignment(scores), patches)


class SelectiveRepresentationSpace(nn.Module):
    """Selective Patching followed by Dynamic Reassembly."""
    def __init__(self, patch_len: int, d_model: int, hidden_size: int, alpha: float, dropout: float, pos: bool, patch_count: int) -> None:
        super().__init__()
        self.select = SelectivePatching(patch_len, d_model, hidden_size, alpha, dropout)
        self.reassemble = DynamicReassembly(alpha)
        self.position = nn.Parameter(torch.zeros(1, 1, patch_count, d_model)) if pos else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected, scores = self.select(patches)
        if self.position is not None:
            selected = selected + self.position
        return self.norm(self.reassemble(selected, scores)), scores


SRS = SelectiveRepresentationSpace


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, features: str = "M", d_model: int = 512,
                 patch_len: int = 24, stride: int = 24, hidden_size: int = 128, dropout: float = 0.2,
                 head_dropout: float = 0.1, alpha: float = 2.0, pos: bool = True,
                 head_mode: str = "linear", affine: bool = True, subtract_last: bool = False) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("SRSNet requires patch_len <= seq_len")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_len, self.stride = patch_len, stride
        patch_count = math.floor((seq_len - patch_len) / stride) + 1
        self.revin = RevIN(enc_in, affine=affine, subtract_last=subtract_last)
        self.srs = SelectiveRepresentationSpace(patch_len, d_model, hidden_size, alpha, dropout, pos, patch_count)
        self.head = FlattenForecastHead(False, enc_in, d_model * patch_count, pred_len, head_dropout)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f"SRSNet expects [B, {self.seq_len}, C]")
        values = self.revin(x_enc, "norm").transpose(1, 2)
        patches = values.unfold(-1, self.patch_len, self.stride)
        representation, _ = self.srs(patches)
        forecast = self.head(representation.transpose(-1, -2)).transpose(1, 2)
        return self.revin(forecast, "denorm")

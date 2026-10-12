"""Clean-room MGSFformer from the published three-module description."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class ResidualDeRedundant(nn.Module):
    """Remove information predictable from the next finer granularity."""
    def __init__(self, width: int) -> None:
        super().__init__()
        self.predict_redundancy = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, coarse: torch.Tensor, finer: torch.Tensor) -> torch.Tensor:
        return self.norm(coarse - self.predict_redundancy(finer))


class SpatioTemporalAttention(nn.Module):
    """Temporal attention per station followed by spatial attention per step."""
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.spatial = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(width)
        self.norm_s = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        temporal = x.transpose(1, 2).reshape(b * n, t, d)
        temporal = self.norm_t(temporal + self.temporal(temporal, temporal, temporal, need_weights=False)[0])
        spatial = temporal.reshape(b, n, t, d).transpose(1, 2).reshape(b * t, n, d)
        spatial = self.norm_s(spatial + self.spatial(spatial, spatial, spatial, need_weights=False)[0])
        return spatial.reshape(b, t, n, d)


class DynamicFusion(nn.Module):
    """Sample/node-specific convex fusion of granularity forecasts."""
    def __init__(self, width: int, branches: int) -> None:
        super().__init__()
        self.score = nn.Linear(width, 1)
        self.branches = branches
        self.last_weights: torch.Tensor | None = None

    def forward(self, features: list[torch.Tensor], forecasts: list[torch.Tensor]) -> torch.Tensor:
        scores = torch.stack([self.score(value.mean(1)).squeeze(-1) for value in features], -1)
        weights = scores.softmax(-1)
        self.last_weights = weights
        return (weights[:, None] * torch.stack(forecasts, -1)).sum(-1)


class Model(nn.Module):
    """Multi-granularity spatiotemporal fusion Transformer."""
    GRANULARITIES = (1, 3, 6, 12, 24)

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, IE_dim: int = 32,
                 dropout: float = 0.3, num_head: int = 2) -> None:
        super().__init__()
        if seq_len < 24 or seq_len % 24:
            raise ValueError("MGSFformer requires seq_len to be a multiple of 24")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in)
        self.embed = nn.Linear(1, IE_dim)
        self.deredundant = nn.ModuleList(ResidualDeRedundant(IE_dim) for _ in range(4))
        self.attention = nn.ModuleList(SpatioTemporalAttention(IE_dim, num_head, dropout) for _ in self.GRANULARITIES)
        self.heads = nn.ModuleList(nn.Linear(seq_len * IE_dim, pred_len) for _ in self.GRANULARITIES)
        self.fusion = DynamicFusion(IE_dim, len(self.GRANULARITIES))

    def _granularity(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        b, t, n, d = x.shape
        coarse = x.reshape(b, t // factor, factor, n, d).mean(2)
        return F.interpolate(coarse.permute(0, 2, 3, 1).reshape(b, n * d, t // factor),
                             size=t, mode="linear", align_corners=False).reshape(b, n, d, t).permute(0, 3, 1, 2)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        raw = [self._granularity(self.embed(normalized.unsqueeze(-1)), f) for f in self.GRANULARITIES]
        features = [raw[0]]
        for index in range(1, len(raw)):
            features.append(self.deredundant[index - 1](raw[index], raw[index - 1]))
        features = [block(value) for block, value in zip(self.attention, features)]
        forecasts = [head(value.transpose(1, 2).flatten(2)).transpose(1, 2)
                     for head, value in zip(self.heads, features)]
        return self.revin(self.fusion(features, forecasts), "denorm")

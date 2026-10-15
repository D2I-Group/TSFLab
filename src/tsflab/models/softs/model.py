"""Clean-room SOFTS implementation from the NeurIPS 2024 paper."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.instance_norm import destandardize, standardize


class SeriesCoreFusion(nn.Module):
    """STAR: aggregate all series into one core, then redistribute it."""

    def __init__(self, d_series: int, d_core: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.core_candidates = nn.Sequential(nn.Linear(d_series, d_core), nn.GELU())
        # A scalar bias is unidentifiable before softmax because it shifts every
        # series score by the same amount.  Omitting it avoids a permanently
        # zero-gradient parameter without changing STAR's aggregation.
        self.core_scores = nn.Linear(d_core, 1, bias=False)
        self.redistribute = nn.Sequential(
            nn.Linear(d_series + d_core, d_series),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_series, d_series),
        )

    def aggregate(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.core_candidates(series)
        weights = self.core_scores(candidates).softmax(dim=1)
        return (weights * candidates).sum(dim=1, keepdim=True), weights

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        core, _ = self.aggregate(series)
        return self.redistribute(torch.cat((series, core.expand(-1, series.size(1), -1)), dim=-1))


STAR = SeriesCoreFusion


class SOFTSBlock(nn.Module):
    def __init__(self, d_model: int, d_core: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        activation_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        self.norm_core = nn.LayerNorm(d_model)
        self.core = SeriesCoreFusion(d_model, d_core, dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), activation_layer(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.dropout(self.core(self.norm_core(tokens)))
        return tokens + self.dropout(self.ff(self.norm_ff(tokens)))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        label_len: int = 0,
        d_model: int = 128,
        d_core: int = 64,
        d_ff: int = 256,
        e_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.use_norm = seq_len, pred_len, use_norm
        self.history_embedding = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList(SOFTSBlock(d_model, d_core, d_ff, dropout, activation) for _ in range(e_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.forecast_head = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f"SOFTS expects [B, {self.seq_len}, C]")
        if self.use_norm:
            values, mean, scale = standardize(x_enc, 1e-5, detach_mean=True, detach_std=True)
        else:
            mean, scale, values = 0.0, 1.0, x_enc
        tokens = self.history_embedding(values.transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens)
        forecast = self.forecast_head(self.final_norm(tokens)).transpose(1, 2)
        return destandardize(forecast, mean, scale)

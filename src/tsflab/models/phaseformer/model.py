"""Independent PhaseFormer implementation from the paper's phase equations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class CrossPhaseRouter(nn.Module):
    """Phase-to-router aggregation followed by router-to-phase distribution."""

    def __init__(self, d_model: int, routers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.routers = nn.Parameter(torch.empty(routers, d_model))
        self.aggregate = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.distribute = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        nn.init.normal_(self.routers, std=0.02)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        routers = self.routers.unsqueeze(0).expand(phases.shape[0], -1, -1)
        context, _ = self.aggregate(routers, phases, phases, need_weights=False)
        routed, _ = self.distribute(phases, context, context, need_weights=False)
        phases = self.norm1(phases + routed)
        return self.norm2(phases + self.ffn(phases))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 16,
        period: int = 24,
        num_routers: int = 4,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period = period
        self.input_periods = math.ceil(seq_len / period)
        self.output_periods = math.ceil(pred_len / period)
        self.use_revin = use_revin
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.input_periods, d_model)
        self.position = nn.Parameter(torch.zeros(period, d_model))
        self.layers = nn.ModuleList(
            [CrossPhaseRouter(d_model, num_routers, num_heads, dropout) for _ in range(num_layers)]
        )
        self.predictor = nn.Linear(d_model, self.output_periods)
        nn.init.normal_(self.position, std=0.02)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        total = self.input_periods * self.period
        if total != self.seq_len:
            repeats = math.ceil(total / self.seq_len)
            x = x.repeat(1, repeats, 1)[:, -total:]
        return x.reshape(x.shape[0], self.input_periods, self.period, self.enc_in).permute(0, 3, 2, 1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        if self.use_revin:
            x_enc = self.revin(x_enc, "norm")
        tokens = self._tokenize(x_enc).flatten(0, 1)
        phases = self.embedding(tokens) + self.position
        for layer in self.layers:
            phases = layer(phases)
        predicted = self.predictor(phases)
        output = predicted.reshape(x_enc.shape[0], self.enc_in, self.period, self.output_periods)
        output = output.permute(0, 3, 2, 1).reshape(x_enc.shape[0], -1, self.enc_in)
        output = output[:, : self.pred_len]
        if self.use_revin:
            output = self.revin(output, "denorm")
        return output

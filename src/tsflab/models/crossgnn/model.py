"""Clean-room CrossGNN implementation from the NeurIPS paper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.last_value_center import center_on_last_value, restore_last_value


def dominant_periods(values: torch.Tensor, count: int) -> list[int]:
    """AMSI Eqs. 1--3: batch/global FFT amplitudes to integer periods."""
    spectrum = torch.fft.rfft(values, dim=1).abs().mean(dim=(0, 2))
    available = spectrum.numel() - 1
    if count < 1 or count > available:
        raise ValueError("scale_number exceeds available non-DC frequencies")
    spectrum = spectrum.clone()
    spectrum[0] = -torch.inf
    frequencies = torch.topk(spectrum, count).indices
    return [max(1, math.ceil(values.shape[1] / int(index))) for index in frequencies]


class AdaptiveMultiScaleIdentifier(nn.Module):
    """AMSI period-wise average pooling and concatenation (Eqs. 4--5)."""

    def __init__(self, scale_number: int) -> None:
        super().__init__()
        self.scale_number = scale_number
        self.last_periods: list[int] = []
        self.last_lengths: list[int] = []

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, list[int], list[int]]:
        periods = dominant_periods(values, self.scale_number)
        scales = []
        for period in periods:
            pooled = F.avg_pool1d(
                values.transpose(1, 2), kernel_size=period, stride=period
            ).transpose(1, 2)
            scales.append(pooled)
        lengths = [scale.shape[1] for scale in scales]
        self.last_periods, self.last_lengths = periods, lengths
        return torch.cat(scales, dim=1), periods, lengths


def _renormalize(scores: torch.Tensor, retained: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~retained, -torch.inf)
    return torch.softmax(masked, dim=-1)


class SparseCrossGraphLayer(nn.Module):
    """Scale-sensitive temporal and signed variable message passing (Eqs. 6--13)."""

    def __init__(
        self,
        max_time_nodes: int,
        channels: int,
        hidden: int,
        time_embedding: int,
        variable_embedding: int,
        neighbors: int,
        dropout: float,
        use_time_graph: bool,
        use_variable_graph: bool,
    ) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.use_time_graph = use_time_graph
        self.use_variable_graph = use_variable_graph
        if use_time_graph:
            self.time_source = nn.Parameter(torch.randn(max_time_nodes, time_embedding) * 0.02)
            self.time_target = nn.Parameter(torch.randn(time_embedding, max_time_nodes) * 0.02)
            self.time_update = nn.Linear(2 * hidden, hidden)
        if use_variable_graph:
            self.variable_source = nn.Parameter(torch.randn(channels, variable_embedding) * 0.02)
            self.variable_target = nn.Parameter(torch.randn(variable_embedding, channels) * 0.02)
            self.variable_update = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def temporal_adjacency(self, periods: list[int], lengths: list[int]) -> torch.Tensor:
        total = sum(lengths)
        # Softplus is a smooth positive relaxation of the paper's ReLU scores;
        # it avoids permanently dead graph parameters at initialization.
        scores = F.softplus(self.time_source[:total] @ self.time_target[:, :total])
        retained = torch.zeros_like(scores, dtype=torch.bool)
        start = 0
        for period, length in zip(periods, lengths):
            width = min(length, max(1, math.ceil(self.neighbors / period)))
            segment = scores[:, start : start + length]
            chosen = torch.topk(segment, width, dim=-1).indices + start
            retained.scatter_(1, chosen, True)
            for local in range(length):
                row = start + local
                retained[row, start + max(0, local - 1) : start + min(length, local + 2)] = True
            start += length
        return _renormalize(scores, retained)

    def variable_adjacency(self) -> torch.Tensor:
        scores = F.softplus(self.variable_source @ self.variable_target)
        width = min(self.neighbors, scores.shape[0])
        positive_indices = torch.topk(scores, width, dim=-1).indices
        negative_indices = torch.topk(scores, width, dim=-1, largest=False).indices
        positive_mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, positive_indices, True)
        negative_mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, negative_indices, True)
        positive = _renormalize(scores, positive_mask)
        inverse = 1.0 / (scores + 1e-4)
        negative = -_renormalize(inverse, negative_mask)
        return positive + negative

    def forward(
        self, hidden: torch.Tensor, periods: list[int], lengths: list[int]
    ) -> torch.Tensor:
        if self.use_time_graph:
            temporal = self.temporal_adjacency(periods, lengths)
            message = torch.einsum("ij,bjdc->bidc", temporal, hidden)
            hidden = F.normalize(
                F.gelu(self.time_update(torch.cat((hidden, message), dim=-1))), dim=-1
            )
        if self.use_variable_graph:
            variable = self.variable_adjacency()
            message = torch.einsum("ij,btjc->btic", variable, hidden)
            hidden = F.normalize(
                F.gelu(self.variable_update(torch.cat((hidden, message), dim=-1))), dim=-1
            )
        return self.dropout(hidden)


class Model(nn.Module):
    """CrossGNN with AMSI, Cross-Scale GNN, Cross-Variable GNN, and DMS head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        e_layers: int = 2,
        anti_ood: bool = True,
        tk: int = 3,
        scale_number: int = 4,
        use_tgcn: bool = True,
        use_ngcn: bool = True,
        dropout: float = 0.1,
        tvechidden: int = 8,
        nvechidden: int = 8,
        hidden: int = 16,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("CrossGNN dimensions must be positive")
        if scale_number > seq_len // 2:
            raise ValueError("scale_number exceeds available non-DC frequencies")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.anti_ood = anti_ood
        self.amsi = AdaptiveMultiScaleIdentifier(scale_number)
        self.expansion = nn.Linear(1, hidden)
        self.layers = nn.ModuleList(
            [
                SparseCrossGraphLayer(
                    scale_number * seq_len,
                    enc_in,
                    hidden,
                    tvechidden,
                    nvechidden,
                    tk,
                    dropout,
                    use_tgcn,
                    use_ngcn,
                )
                for _ in range(e_layers)
            ]
        )
        self.channel_head = nn.Linear(hidden, 1)
        self.temporal_head = nn.Linear(seq_len, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(
                f"x_enc must have shape (batch, {self.seq_len}, {self.channels})"
            )
        if self.anti_ood:
            centered, baseline = center_on_last_value(x_enc)
        else:
            centered, baseline = x_enc, 0.0
        multiscale, periods, lengths = self.amsi(centered)
        hidden = self.expansion(multiscale.unsqueeze(-1))
        for layer in self.layers:
            hidden = hidden + layer(hidden, periods, lengths)
        collapsed = self.channel_head(hidden).squeeze(-1).transpose(1, 2)
        # AMSI has data-dependent L'; interpolation makes the DMS head shape-static.
        collapsed = F.interpolate(
            collapsed, size=self.seq_len, mode="linear", align_corners=False
        )
        forecast = self.temporal_head(collapsed).transpose(1, 2)
        return restore_last_value(forecast, baseline)

"""Local StemGNN implementation from paper and official-code review."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class LatentCorrelationGraph(nn.Module):
    """Infer a symmetric latent graph from node history embeddings."""

    def __init__(self, seq_len: int, hidden: int, dropout: float, slope: float) -> None:
        super().__init__()
        self.history_encoder = nn.GRU(1, hidden, batch_first=True)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, nodes = x.shape
        encoded, _ = self.history_encoder(x.transpose(1, 2).reshape(batch * nodes, length, 1))
        summary = encoded[:, -1].reshape(batch, nodes, -1)
        scores = self.activation(self.query(summary) @ self.key(summary).transpose(-1, -2))
        directed = torch.softmax(scores, -1)
        graph = 0.5 * (directed + directed.transpose(-1, -2))
        return self.dropout(graph)


def _chebyshev_graph_terms(x: torch.Tensor, graph: torch.Tensor, order: int = 4) -> torch.Tensor:
    degree = graph.sum(-1).clamp_min(1e-6)
    normalized = graph * degree.rsqrt().unsqueeze(-1) * degree.rsqrt().unsqueeze(-2)
    laplacian = torch.eye(graph.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0) - normalized
    basis = [torch.eye(graph.shape[-1], device=x.device, dtype=x.dtype).expand(graph.shape[0], -1, -1)]
    if order > 1:
        basis.append(laplacian)
    for _ in range(2, order):
        basis.append(2 * laplacian @ basis[-1] - basis[-2])
    return torch.stack([torch.einsum("bnm,blm->bln", support, x) for support in basis], 1)


class SpectralTemporalBlock(nn.Module):
    """Joint graph-Chebyshev and temporal-Fourier filtering block."""

    def __init__(self, seq_len: int, multiplier: int, with_backcast: bool) -> None:
        super().__init__()
        frequencies = seq_len // 2 + 1
        self.seq_len = seq_len
        self.weight = nn.Parameter(torch.empty(4, multiplier, frequencies, 2))
        self.channel_mix = nn.Linear(multiplier, 1)
        self.forecast = nn.Linear(seq_len, seq_len)
        self.backcast = nn.Linear(seq_len, seq_len) if with_backcast else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        terms = _chebyshev_graph_terms(x, graph)
        spectrum = torch.fft.rfft(terms, dim=2)
        weight = torch.view_as_complex(self.weight.contiguous()).view(1, 4, -1, spectrum.shape[2], 1)
        filtered = torch.fft.irfft(spectrum.unsqueeze(2) * weight, n=self.seq_len, dim=3).sum(1)
        filtered = self.channel_mix(filtered.permute(0, 2, 3, 1)).squeeze(-1)
        forecast = self.forecast(filtered.transpose(1, 2)).transpose(1, 2)
        backcast = None if self.backcast is None else self.backcast(filtered.transpose(1, 2)).transpose(1, 2)
        return forecast, backcast


class Model(nn.Module):
    """Latent-graph spectral-temporal forecaster with residual stacks."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        multi_layer: int = 3,
        dropout_rate: float = 0.5,
        leaky_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.graph = LatentCorrelationGraph(seq_len, num_nodes, dropout_rate, leaky_rate)
        self.blocks = nn.ModuleList(
            [
                SpectralTemporalBlock(seq_len, multi_layer, True),
                SpectralTemporalBlock(seq_len, multi_layer, False),
            ]
        )
        self.horizon = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.LeakyReLU(leaky_rate), nn.Linear(seq_len, pred_len)
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc
        check_window(x_enc, self.seq_len, self.num_nodes)
        graph = self.graph(x_enc)
        residual = x_enc
        forecasts = []
        for block in self.blocks:
            forecast, backcast = block(residual, graph)
            forecasts.append(forecast)
            if backcast is not None:
                residual = residual - backcast
        combined = torch.stack(forecasts).sum(0)
        return self.horizon(combined.transpose(1, 2)).transpose(1, 2)


__all__ = ["Model", "LatentCorrelationGraph", "SpectralTemporalBlock"]

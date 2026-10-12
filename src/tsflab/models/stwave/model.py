"""Clean-room STWave from arXiv:2112.02740.

The implementation uses a differentiable two-band discrete wavelet split,
dual temporal/spatial encoders, Laplacian spectral positional encoding and a
query-sampled graph attention mask.  It does not contain BasicTS source.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


def _graph_data(adjacency: np.ndarray, nodes: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = square_adjacency(adjacency, nodes)
    symmetric = np.maximum(matrix, matrix.T) + np.eye(nodes, dtype=np.float32)
    degree = symmetric.sum(-1)
    normalized = symmetric / np.sqrt(np.maximum(degree[:, None] * degree[None, :], 1e-8))
    laplacian = np.eye(nodes, dtype=np.float32) - normalized
    _, vectors = np.linalg.eigh(laplacian)
    take = min(width, nodes)
    spectral = np.zeros((nodes, width), dtype=np.float32)
    spectral[:, :take] = vectors[:, :take]
    return torch.from_numpy(matrix > 0), torch.from_numpy(spectral)


def wavelet_disentangle(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-level orthogonal Haar reconstruction into low/high bands."""
    original = x.shape[1]
    if original == 1:
        return x, torch.zeros_like(x)
    values = x.transpose(1, 2)
    if original % 2:
        values = F.pad(values, (0, 1), mode="replicate")
    even, odd = values[..., 0::2], values[..., 1::2]
    low_coeff = (even + odd) / math.sqrt(2.0)
    high_coeff = (even - odd) / math.sqrt(2.0)
    low = torch.stack((low_coeff, low_coeff), -1).flatten(-2)[..., :original] / math.sqrt(2.0)
    high = torch.stack((high_coeff, -high_coeff), -1).flatten(-2)[..., :original] / math.sqrt(2.0)
    return low.transpose(1, 2), high.transpose(1, 2)


class SpectralGraphAttention(nn.Module):
    """Sparse neighbor attention; sampled high-energy queries also attend globally."""
    def __init__(self, width: int, adjacency: torch.Tensor, spectral: torch.Tensor,
                 log_samples: int) -> None:
        super().__init__()
        self.query, self.key, self.value, self.output = (nn.Linear(width, width) for _ in range(4))
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("spectral", spectral)
        self.log_samples = log_samples
        self.norm = nn.LayerNorm(width)
        self.last_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        positioned = x + self.spectral[None, None]
        q, k, v = self.query(positioned), self.key(positioned), self.value(positioned)
        logits = torch.einsum("btnd,btmd->btnm", q, k) / math.sqrt(d)
        base = self.adjacency | torch.eye(n, dtype=torch.bool, device=x.device)
        energy = q.square().mean(-1)
        sampled = max(1, min(n, int(round(self.log_samples * math.log2(max(n, 2))))))
        query_indices = energy.topk(sampled, -1).indices
        mask = base[None, None].expand(b, t, -1, -1).clone()
        mask.scatter_(2, query_indices[..., None].expand(-1, -1, -1, n), True)
        attention = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max).softmax(-1)
        self.last_mask = mask
        result = torch.einsum("btnm,btmd->btnd", attention, v)
        return self.norm(x + self.output(result))


class TemporalModule(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 1, batch_first=True)
        self.causal = nn.Conv1d(width, width, 3, padding=2)
        self.gate = nn.Conv1d(width, width, 3, padding=2)
        self.norm = nn.LayerNorm(width)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        flat = x.transpose(1, 2).reshape(b * n, t, d)
        attended = self.attention(flat, flat, flat, need_weights=False)[0]
        channels = flat.transpose(1, 2)
        causal = torch.tanh(self.causal(channels)[..., :t]) * torch.sigmoid(self.gate(channels)[..., :t])
        return self.norm(flat + attended + causal.transpose(1, 2)).reshape(b, n, t, d).transpose(1, 2)


class DualEncoder(nn.Module):
    def __init__(self, width: int, adjacency: torch.Tensor, spectral: torch.Tensor,
                 log_samples: int) -> None:
        super().__init__()
        self.temporal = TemporalModule(width)
        self.spatial = SpectralGraphAttention(width, adjacency, spectral, log_samples)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.temporal(x))


class AdaptiveFusion(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(3 * width, width), nn.Sigmoid())
        self.last_gate: torch.Tensor | None = None
    def forward(self, low: torch.Tensor, high: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat((low, high, calendar), -1))
        self.last_gate = gate
        return gate * low + (1 - gate) * high


class Model(nn.Module):
    """Wavelet-disentangled efficient spectral graph attention forecaster."""
    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx=None,
                 hidden_size: int = 16, layers: int = 1,
                 log_samples: int = 1) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("STWave dimensions must be positive")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        if adj_mx is None:
            adj_mx = np.eye(num_nodes, dtype=np.float32)
            if num_nodes > 1:
                for index in range(num_nodes):
                    adj_mx[index, (index + 1) % num_nodes] = 1
        adjacency, spectral = _graph_data(adj_mx, num_nodes, hidden_size)
        self.value_embedding = nn.Linear(1, hidden_size)
        self.calendar_embedding = nn.Linear(2, hidden_size)
        self.low_encoders = nn.ModuleList(DualEncoder(hidden_size, adjacency, spectral, log_samples) for _ in range(layers))
        self.high_encoders = nn.ModuleList(DualEncoder(hidden_size, adjacency, spectral, log_samples) for _ in range(layers))
        self.low_head = nn.Linear(seq_len * hidden_size, pred_len * hidden_size)
        self.high_head = nn.Linear(seq_len * hidden_size, pred_len * hidden_size)
        self.future_calendar = nn.Linear(seq_len * hidden_size, pred_len * hidden_size)
        self.fusion = AdaptiveFusion(hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.num_nodes)
        values = to_spatiotemporal(x_enc, x_mark_enc)
        calendar = values[..., 1:3]
        if calendar.shape[-1] < 2:
            calendar = F.pad(calendar, (0, 2 - calendar.shape[-1]))
        calendar = self.calendar_embedding(calendar)
        low, high = wavelet_disentangle(x_enc)
        low, high = self.value_embedding(low.unsqueeze(-1)) + calendar, self.value_embedding(high.unsqueeze(-1)) + calendar
        for low_encoder, high_encoder in zip(self.low_encoders, self.high_encoders):
            low, high = low_encoder(low), high_encoder(high)
        b, _, n, d = low.shape
        low = self.low_head(low.transpose(1, 2).flatten(2)).reshape(b, n, self.pred_len, d).transpose(1, 2)
        high = self.high_head(high.transpose(1, 2).flatten(2)).reshape(b, n, self.pred_len, d).transpose(1, 2)
        future = self.future_calendar(calendar.transpose(1, 2).flatten(2)).reshape(b, n, self.pred_len, d).transpose(1, 2)
        return self.readout(self.fusion(low, high, future)).squeeze(-1)

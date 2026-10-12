"""Clean-room MAGE implementation from the NeurIPS 2025 method description.

Adaptive graph experts use factorised kernels, so propagation stays linear in
the node count and never materialises a learned ``N x N`` adjacency.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from tsflab.models._components.marks import to_calendar_spatiotemporal
from tsflab.models._inputs import check_window


class AdaptiveGraphExpert(nn.Module):
    """Low-rank kernel propagation ``softmax(E1) softmax(E2) X``."""
    def __init__(self, nodes: int, width: int, rank: int) -> None:
        super().__init__()
        self.source = nn.Parameter(torch.randn(nodes, rank) / math.sqrt(rank))
        self.target = nn.Parameter(torch.randn(nodes, rank) / math.sqrt(rank))
        self.value = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.einsum("nr,bnd->brd", self.source.softmax(0), self.value(x))
        return torch.einsum("nr,brd->bnd", self.target.softmax(-1), pooled)


class MixtureGraphBlock(nn.Module):
    """Sparse, load-balanced mixture of adaptive graph experts."""
    def __init__(self, nodes: int, width: int, experts: int, topk: int, rank: int) -> None:
        super().__init__()
        self.topk = min(topk, experts)
        self.norm = nn.RMSNorm(width)
        self.router = nn.Linear(width, experts)
        self.experts = nn.ModuleList(AdaptiveGraphExpert(nodes, width, rank) for _ in range(experts))
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))
        self.last_routing: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        logits = self.router(h)
        values, indices = logits.topk(self.topk, dim=-1)
        sparse = logits.new_full(logits.shape, float("-inf"))
        sparse.scatter_(-1, indices, values)
        routing = sparse.softmax(-1)
        # The low-weight dense path keeps expert utilization balanced/trainable.
        routing = 0.95 * routing + 0.05 * logits.softmax(-1).mean((0, 1), keepdim=True)
        expert_values = torch.stack([expert(h) for expert in self.experts], dim=-2)
        self.last_routing = routing
        x = x + (routing.unsqueeze(-1) * expert_values).sum(-2)
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    """Mixture of Adaptive Graph Experts forecaster."""
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, model_dim: int = 64,
                 recur_num: int = 8, topk: int = 2, node_dim: int = 16) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.history = nn.Linear(seq_len * 3, model_dim)
        self.calendar = nn.Sequential(nn.Linear(2, model_dim), nn.SiLU())
        self.blocks = nn.ModuleList(MixtureGraphBlock(enc_in, model_dim, recur_num, topk, node_dim) for _ in range(3))
        self.forecast = nn.Linear(model_dim, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(x_enc.shape[0], self.seq_len, 6)
        values = to_calendar_spatiotemporal(x_enc, x_mark_enc)
        h = self.history(values.transpose(1, 2).flatten(2))
        h = h + self.calendar(values[..., 1:3].mean(1))
        skip = h
        for depth, block in enumerate(self.blocks, start=1):
            update = h
            for _ in range(depth):
                update = block(update)
            h = update if depth < 3 else skip - update
        return self.forecast(h + skip).transpose(1, 2)

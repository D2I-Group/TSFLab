"""Independent TimeFilter implementation from the ICML 2025 paper."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class PatchGraphBuilder(nn.Module):
    """Build a graph whose nodes are individual channel/patch pairs."""
    def __init__(self, patch_len, width):
        super().__init__()
        self.embedding = nn.Linear(patch_len, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.scale = width ** -0.5

    def forward(self, x):
        # x (B,C,L) -> nodes (B,C,P,D)
        patches = math.ceil(x.shape[-1] / self.embedding.in_features)
        x = F.pad(x, (0, patches * self.embedding.in_features - x.shape[-1]))
        nodes = self.embedding(x.unfold(-1, self.embedding.in_features, self.embedding.in_features))
        flat = nodes.flatten(1, 2)
        affinity = (self.query(flat) @ self.key(flat).transpose(-1, -2)) * self.scale
        return nodes, affinity


class RegionExpert(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.message = nn.Linear(width, width)
        self.update = nn.Sequential(nn.Linear(2 * width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, nodes, adjacency):
        messages = adjacency @ self.message(nodes)
        return self.update(torch.cat((nodes, messages), -1))


class PatchSpecificGraphFilter(nn.Module):
    """MoE router filters graph regions separately for every patch node."""
    def __init__(self, width, hidden, experts, top_p, dropout):
        super().__init__()
        self.top_p = top_p
        self.router = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, experts))
        self.experts = nn.ModuleList(RegionExpert(width, hidden, dropout) for _ in range(experts))
        self.norm = nn.LayerNorm(width)
        self.last_adjacency = None
        self.last_routes = None
        self.last_moe_loss = None

    def forward(self, nodes, logits):
        probabilities = logits.softmax(-1)
        count = max(1, round(probabilities.shape[-1] * self.top_p))
        threshold = probabilities.topk(count, -1).values[..., -1:]
        retained = probabilities * (probabilities >= threshold)
        adjacency = retained / retained.sum(-1, keepdim=True).clamp_min(1e-6)
        routes = self.router(nodes).softmax(-1)
        expert_outputs = torch.stack([expert(nodes, adjacency) for expert in self.experts], -2)
        update = (expert_outputs * routes.unsqueeze(-1)).sum(-2)
        self.last_adjacency, self.last_routes = adjacency, routes
        load = routes.mean((0, 1))
        self.last_moe_loss = (load * load).sum() * len(self.experts)
        return self.norm(nodes + update)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, label_len=0, features="M",
                 d_model=64, d_ff=128, e_layers=2, patch_len=16,
                 dropout=0.1, top_p=0.5, pos=True, num_experts=4):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len = patch_len
        self.patch_count = math.ceil(seq_len / patch_len)
        self.revin = RevIN(enc_in)
        self.graph_builder = PatchGraphBuilder(patch_len, d_model)
        self.position = nn.Parameter(torch.randn(1, enc_in, self.patch_count, d_model) * 0.02) if pos else None
        self.filters = nn.ModuleList(PatchSpecificGraphFilter(d_model, d_ff, num_experts, top_p, dropout) for _ in range(e_layers))
        self.head = nn.Sequential(nn.Flatten(-2), nn.Linear(self.patch_count * d_model, pred_len))

    @property
    def last_moe_loss(self):
        losses = [layer.last_moe_loss for layer in self.filters if layer.last_moe_loss is not None]
        return sum(losses) / len(losses) if losses else None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm").transpose(1, 2)
        patches, affinity = self.graph_builder(normalized)
        if self.position is not None:
            patches = patches + self.position
        nodes = patches.flatten(1, 2)
        for graph_filter in self.filters:
            nodes = graph_filter(nodes, affinity)
            affinity = (nodes @ nodes.transpose(-1, -2)) / math.sqrt(nodes.shape[-1])
        patches = nodes.reshape(x_enc.shape[0], self.enc_in, self.patch_count, -1)
        return self.revin(self.head(patches).transpose(1, 2), "denorm")

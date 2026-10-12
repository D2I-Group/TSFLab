"""Ada-MSHyper (Shang et al., NeurIPS 2024, arXiv:2410.23992).

Independent implementation from the paper's equations. Pipeline: RevIN, a
multi-scale feature extraction pyramid (strided conv aggregation), per-scale
adaptive hypergraph learning with node/hyperedge constraint losses, intra-scale
hypergraph convolution attention, inter-scale hyperedge self-attention, and a
linear prediction layer over the concatenated node and hyperedge tokens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.revin import RevIN

_NEG = -1e9


class ScaleHypergraph(nn.Module):
    """Adaptive hypergraph learning and intra-scale convolution at one scale.

    Input ``[B, N, D]`` node series; returns updated node features ``[B, N, D]``,
    hyperedge features ``[B, M, D]`` (all-zero rows for empty hyperedges), the
    non-empty hyperedge mask ``[M]``, and this scale's node and hyperedge
    constraint losses.
    """

    def __init__(
        self, num_nodes: int, num_edges: int, channels: int, d_embed: int,
        eta: int, beta: float, gamma: float, dropout: float,
    ) -> None:
        super().__init__()
        self.eta, self.beta, self.gamma = eta, beta, gamma
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, d_embed))
        self.hyper_embedding = nn.Parameter(torch.randn(num_edges, d_embed))
        self.node_init = nn.Linear(channels, channels, bias=False)  # f in V = f(X)
        self.attn_node = nn.Linear(channels, 1, bias=False)  # f_t on [v_i, e_j] split in two halves
        self.attn_edge = nn.Linear(channels, 1, bias=False)
        self.weight = nn.Linear(channels, channels, bias=False)  # P in Eq. 15
        self.dropout = nn.Dropout(dropout)

    def incidence(self) -> torch.Tensor:
        """Eq. 4-6: softmax(relu(E_node E_hyper^T)), top-eta per node, threshold beta.

        The forward value is the binary matrix; a straight-through term lets the
        soft scores receive gradients through the attention-enriched incidence.
        """
        soft = adaptive_node_embedding_adjacency(self.node_embedding, self.hyper_embedding.t())
        top = soft.topk(min(self.eta, soft.shape[1]), dim=1).indices
        keep = torch.zeros_like(soft).scatter_(1, top, 1.0) * (soft.detach() > self.beta).to(soft.dtype)
        return keep + (soft - soft.detach()) * keep

    def forward(self, x: torch.Tensor):
        H = self.incidence()  # [N, M]
        member = (H.detach() > 0.5)
        deg_e = H.detach().sum(0)  # [M]
        deg_n = H.detach().sum(1)  # [N]
        valid = (deg_e > 0).to(x.dtype)  # non-empty hyperedges
        v = self.node_init(x)  # [B, N, D]
        # Eq. 7: hyperedge = mean of its member nodes.
        edges = torch.einsum("nm,bnd->bmd", H, v) / deg_e.clamp(min=1.0).view(1, -1, 1)

        # Eq. 8-9: node constraint, mean |v_j - e_i| over a node's hyperedges, then over nodes.
        gap = (v.unsqueeze(2) - edges.unsqueeze(1)).abs().mean(-1)  # [B, N, M]
        per_node = (gap * H).sum(2) / deg_n.clamp(min=1.0)
        node_loss = per_node.mean(1).mean()

        # Eq. 10-12: hyperedge constraint over non-empty hyperedges.
        unit = F.normalize(edges, dim=-1, eps=1e-8)
        alpha = unit @ unit.transpose(1, 2)  # cosine similarity [B, M, M]
        dist = ((edges.unsqueeze(2) - edges.unsqueeze(1)).square().sum(-1) + 1e-12).sqrt()
        pair = valid.view(-1, 1) * valid.view(1, -1)
        term = alpha * dist + (1.0 - alpha) * F.relu(self.gamma - dist)
        count = valid.sum().clamp(min=1.0)
        hyper_loss = ((term * pair).sum((1, 2)) / (count * count)).mean()

        # Eq. 14: attention of each node over its hyperedges.
        score = F.leaky_relu(self.attn_node(v) + self.attn_edge(edges).transpose(1, 2), 0.2)
        score = score.masked_fill(~member.unsqueeze(0), _NEG)
        enriched = self.dropout(torch.softmax(score, dim=-1)) * H.unsqueeze(0)  # [B, N, M]

        # Eq. 15: sigma(Dv^-1/2 H De^-1 H^T Dv^-1/2 V P) with the enriched incidence.
        dv = enriched.sum(2)
        de = enriched.sum(1)
        dv_inv_sqrt = dv.clamp(min=1e-6).pow(-0.5) * (dv > 1e-6).to(x.dtype)
        de_inv = de.clamp(min=1e-6).reciprocal() * (de > 1e-6).to(x.dtype)
        h = dv_inv_sqrt.unsqueeze(-1) * self.weight(v)
        h = torch.einsum("bnm,bnd->bmd", enriched, h) * de_inv.unsqueeze(-1)
        h = torch.einsum("bnm,bmd->bnd", enriched, h) * dv_inv_sqrt.unsqueeze(-1)
        return F.leaky_relu(h, 0.2), edges, valid, node_loss, hyper_loss


class ScaleConv(nn.Module):
    """Eq. 3 aggregation: strided Conv1d, BatchNorm, ELU, window = stride."""

    def __init__(self, channels: int, window: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, window, stride=window)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D, N] -> [B, D, N // w]
        return F.elu(self.norm(self.conv(x)))


def _scale_nodes(seq_len: int, window_size: tuple) -> list[int]:
    """Node count of every scale: ``seq_len`` floor-divided by the windows in turn."""
    nodes = [seq_len]
    for w in window_size:
        nodes.append(nodes[-1] // w)
    if nodes[-1] < 1:
        raise ValueError("seq_len is too short for the aggregation windows")
    return nodes


class Model(nn.Module):
    """AdaMSHyper forecaster: [B, L, C] history -> [B, H, C] forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        window_size: Sequence[int] = (4, 4),
        hyper_num: Sequence[int] = (50, 20, 10),
        d_embed: int = 16,
        eta: int = 3,
        beta: float = 0.5,
        gamma: float = 4.2,
        lambda_balance: float = 0.5,
        const_weight: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        window_size, hyper_num = tuple(window_size), tuple(hyper_num)
        if min(seq_len, pred_len) < 1:
            raise ValueError("invalid AdaMSHyper dimension")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.lambda_balance, self.const_weight = lambda_balance, const_weight
        nodes = _scale_nodes(seq_len, window_size)
        self.num_nodes = tuple(nodes)
        self.hyper_num = hyper_num

        self.revin = RevIN(enc_in, affine=False)
        self.aggregators = nn.ModuleList(ScaleConv(enc_in, w) for w in window_size)
        self.scales = nn.ModuleList(
            ScaleHypergraph(n, m, enc_in, d_embed, eta, beta, gamma, dropout)
            for n, m in zip(nodes, hyper_num)
        )
        self.query = nn.Linear(enc_in, enc_in)
        self.key = nn.Linear(enc_in, enc_in)
        self.value = nn.Linear(enc_in, enc_in)
        self.head = nn.Linear(sum(nodes) + sum(hyper_num), pred_len)
        self.aux_loss: torch.Tensor | None = None

    def multi_scale(self, x: torch.Tensor) -> list[torch.Tensor]:
        """[B, L, D] -> list of [B, N_s, D]; scale 1 is the input itself."""
        series = [x]
        current = x.transpose(1, 2)
        for aggregate in self.aggregators:
            current = aggregate(current)
            series.append(current.transpose(1, 2))
        return series

    def hyperedge_attention(self, edges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Eq. 16 over all scales' hyperedges; empty hyperedges are neither keys nor outputs."""
        scores = self.query(edges) @ self.key(edges).transpose(1, 2) / math.sqrt(edges.shape[-1])
        scores = scores.masked_fill(valid.view(1, 1, -1) < 0.5, _NEG)
        out = torch.softmax(scores, dim=-1) @ self.value(edges)
        return out * valid.view(1, -1, 1)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in}), got {tuple(x_enc.shape)}")
        series = self.multi_scale(self.revin(x_enc, "norm"))
        nodes, edges, valids = [], [], []
        node_loss = hyper_loss = x_enc.new_zeros(())
        for scale, s in zip(self.scales, series):
            n, e, valid, ln, lh = scale(s)
            nodes.append(n)
            edges.append(e)
            valids.append(valid)
            node_loss, hyper_loss = node_loss + ln, hyper_loss + lh
        mixed = self.hyperedge_attention(torch.cat(edges, 1), torch.cat(valids))
        tokens = torch.cat(nodes + [mixed], dim=1)  # [B, sum(N) + sum(M), D]
        forecast = self.head(tokens.transpose(1, 2)).transpose(1, 2)  # [B, H, D]
        if self.training:
            # Eq. 13; added to the point loss by the trainer as the auxiliary term.
            self.aux_loss = self.const_weight * (
                self.lambda_balance * node_loss + (1.0 - self.lambda_balance) * hyper_loss
            )
        else:
            self.aux_loss = None
        return self.revin(forecast, "denorm")

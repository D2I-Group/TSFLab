"""Differentiable top-k local expert self-attention.

Treats every key/value position as a candidate "expert". For each query, a
lightweight router scores all positions, keeps only the ``topk`` highest
scoring ones (softmax-reweighted), and attention is computed only over that
reduced set -- turning full quadratic self-attention into an adaptive local
gather-then-attend operation. An optional shared global expert (the sequence
mean key/value) is appended to every query's selected set so long-range
context is never fully discarded.
"""

from __future__ import annotations

import torch
from torch import nn


class LocalExpertRouter(nn.Module):
    """Score every (query, key) pair and keep the top-``topk`` keys per query."""

    def __init__(self, qk_dim: int, topk: int, scale: float | None = None) -> None:
        super().__init__()
        self.topk = topk
        self.scale = scale if scale is not None else qk_dim**-0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``query, key: (n, m, c)`` -> ``(weight, index): (n, m, topk)``."""
        logits = (query * self.scale) @ key.transpose(-2, -1)
        topk_logits, index = torch.topk(logits, k=self.topk, dim=-1)
        weight = topk_logits.softmax(dim=-1)
        return weight, index


def gather_experts(index: torch.Tensor, weight: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Gather routed key/value experts and apply their routing weight.

    ``index, weight: (n, m, topk)``, ``kv: (n, m, c)`` -> ``(n, m, topk, c)``.
    """
    n, m, c = kv.shape
    expanded_kv = kv.unsqueeze(1).expand(-1, m, -1, -1)
    expanded_index = index.unsqueeze(-1).expand(-1, -1, -1, c)
    selected = torch.gather(expanded_kv, dim=2, index=expanded_index)
    return weight.unsqueeze(-1) * selected


class TopKExpertAttention(nn.Module):
    """Adaptive local self-attention over the top-``topk`` scoring positions.

    Input/output: ``(batch, tokens, dim)``. A depthwise positional convolution
    is added as a residual before projection, matching the local relative
    position handling used by patch-token forecasting encoders.
    ``topk=0`` is the dense mode (ordinary softmax attention, no router);
    ``topk < 0``, ``num_heads < 1`` and a ``qk_dim`` not divisible by
    ``num_heads`` raise ``ValueError``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        topk: int = 4,
        shared: bool = False,
        qk_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be at least 1, got {num_heads}")
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if topk < 0:
            raise ValueError(f"topk must be non-negative (0 selects dense attention), got {topk}")
        if (qk_dim or dim) % num_heads:
            raise ValueError("qk_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.topk = topk
        self.shared = shared
        qk_dim = qk_dim or dim
        self.qk_dim = qk_dim
        self.head_qk = qk_dim // num_heads
        self.head_v = dim // num_heads

        self.positional = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.qkv = nn.Linear(dim, 2 * qk_dim + dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        if topk > 0:
            self.router = LocalExpertRouter(self.head_qk, topk)
        self.last_route_weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional(x.transpose(1, 2)).transpose(1, 2)
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)

        def split_heads(t: torch.Tensor, head_dim: int) -> torch.Tensor:
            return (
                t.view(batch, tokens, self.num_heads, head_dim)
                .permute(0, 2, 1, 3)
                .reshape(batch * self.num_heads, tokens, head_dim)
            )

        q = split_heads(q, self.head_qk)
        k = split_heads(k, self.head_qk)
        v = split_heads(v, self.head_v)

        if self.shared:
            global_key = k.mean(dim=1, keepdim=True)
            global_value = v.mean(dim=1, keepdim=True)

        if self.topk > 0:
            weight, index = self.router(q, k)
            self.last_route_weight = weight
            kv = torch.cat([k, v], dim=-1)
            selected = gather_experts(index, weight, kv)
            k_sel, v_sel = selected.split([self.head_qk, self.head_v], dim=-1)
            if self.shared:
                k_sel = torch.cat([k_sel, global_key.unsqueeze(2).expand(-1, tokens, -1, -1)], dim=2)
                v_sel = torch.cat([v_sel, global_value.unsqueeze(2).expand(-1, tokens, -1, -1)], dim=2)
            scale = self.head_qk**-0.5
            attn_logits = torch.einsum("nqd,nqkd->nqk", q * scale, k_sel)
            attn_weight = self.attn_dropout(attn_logits.softmax(dim=-1))
            out = torch.einsum("nqk,nqkd->nqd", attn_weight, v_sel)
        else:
            if self.shared:
                k = torch.cat([k, global_key], dim=1)
                v = torch.cat([v, global_value], dim=1)
            scale = self.head_qk**-0.5
            attn_logits = (q * scale) @ k.transpose(-2, -1)
            attn_weight = self.attn_dropout(attn_logits.softmax(dim=-1))
            out = attn_weight @ v

        out = (
            out.reshape(batch, self.num_heads, tokens, self.head_v).permute(0, 2, 1, 3).reshape(batch, tokens, self.dim)
        )
        return self.out_proj(out)

"""Differential self-attention.

Splits queries and keys into two halves, forms two ordinary softmax attention
maps from them, and subtracts a learned, per-head scalar
times the second map from the first before applying the result to values.
Subtracting a second attention map that has learned to track common-mode
("noise") structure cancels attention noise the same way a differential
amplifier cancels common-mode signal, leaving a sparser, sharper attention
pattern. The attended output is re-normalized with RMSNorm and rescaled by
``(1 - lambda_init)`` so its magnitude matches ordinary single-map attention.

The subtraction weight is ``lambda = exp(q1.k1) - exp(q2.k2) + lambda_init``
per head. It is unconstrained: nothing keeps it in ``[0, 1]`` or non-negative,
so it is not a convex combination and may be negative or exceed one.
"""

from __future__ import annotations

import torch
from torch import nn


class DifferentialAttention(nn.Module):
    """Full (non-causal) differential self-attention.

    Input/output: ``queries, keys, values: (batch, seq, heads, 2 * head_dim)``
    for queries/keys and ``(batch, seq, heads, 2 * head_dim)`` for values;
    returns ``(batch, seq, heads, 2 * head_dim)``. Splitting the head
    dimension in two (one half per softmax map) and doubling the value
    dimension mirrors the paper's construction; a wrapping linear projection
    is expected to map back to ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        lambda_init: float = 0.8,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        head_dim = d_model // num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.lambda_init = lambda_init
        self.scale = head_dim**-0.5
        self.dropout = nn.Dropout(attention_dropout)

        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, head_dim) * 0.1)
        self.rms_scale = nn.Parameter(torch.ones(2 * head_dim))
        self.eps = 1e-5

    def _lambda(self) -> torch.Tensor:
        lambda1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1))
        lambda2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1))
        return (lambda1 - lambda2 + self.lambda_init).view(1, -1, 1, 1)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = queries.chunk(2, dim=-1)
        k1, k2 = keys.chunk(2, dim=-1)
        batch, seq_len, heads, _ = queries.shape

        scores1 = torch.softmax(self.scale * torch.einsum("blhe,bshe->bhls", q1, k1), dim=-1)
        scores2 = torch.softmax(self.scale * torch.einsum("blhe,bshe->bhls", q2, k2), dim=-1)
        attn = self.dropout(scores1 - self._lambda() * scores2)
        out = torch.einsum("bhls,bshd->bhld", attn, values)

        flat = out.reshape(batch * heads, seq_len, -1)
        rms = torch.sqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (flat / rms) * self.rms_scale * (1 - self.lambda_init)
        return normalized.view(batch, heads, seq_len, -1).transpose(1, 2)

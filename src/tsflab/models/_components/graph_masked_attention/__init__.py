"""Multi-head attention blending a global dense score matrix with a local
adjacency-masked score matrix.

Implements a node-attention layer that lets learned, data-driven affinity and
a fixed graph adjacency structure shape attention jointly instead of
exclusively. Given queries/keys/values over a node (or any token) axis, this
module computes standard scaled dot-product multi-head attention scores once,
then takes softmax twice from those same raw scores: densely over every token
pair ("global") and, when an ``adj_mask`` is supplied, only over
adjacency-permitted pairs via additive ``-inf`` masking before softmax
("local"). The two probability matrices are averaged before being applied to
``value``:

    scores = (query @ key.transpose(-1, -2)) / sqrt(head_dim)
    global_attn = softmax(scores, dim=-1)
    local_attn  = softmax(scores.masked_fill(~adj_mask, -inf), dim=-1)
    attn = (global_attn + local_attn) / 2  # when adj_mask is given
    out  = attn @ value

When ``adj_mask`` is ``None`` the local term is skipped and this reduces to
plain multi-head self-/cross-attention. This is the paper equation
``GLSAtt = (Softmax(alpha_local) + Softmax(alpha_global)) @ V / 2`` used by
Extralonger's spatial-attention route, generalized so any graph forecaster
that wants to bias full attention with a fixed adjacency (without discarding
long-range, data-driven affinity entirely) can reuse it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalLocalGraphAttention(nn.Module):
    """Multi-head attention averaging global and adjacency-masked local scores.

    Args:
        model_dim: Total feature width of queries/keys/values. Must be
            divisible by ``num_heads``.
        num_heads: Number of attention heads.
    """

    def __init__(self, model_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be at least 1, got {num_heads}")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        adj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply blended global/local attention.

        Args:
            query: Tensor shaped ``(..., L_q, model_dim)``.
            key: Tensor shaped ``(..., L_kv, model_dim)``.
            value: Tensor shaped ``(..., L_kv, model_dim)``.
            adj_mask: Optional boolean tensor broadcastable to
                ``(..., L_q, L_kv)`` where ``True`` marks an allowed
                (adjacent) pair. When ``None``, only the global (dense)
                attention term is used. A query row with no allowed key gets
                an all-zero attention row (its output is ``out_proj`` bias).

        Returns:
            Tensor shaped ``(..., L_q, model_dim)``.
        """
        *batch_shape, len_q, _ = query.shape
        q = self._split_heads(self.fc_q(query))
        k = self._split_heads(self.fc_k(key))
        v = self._split_heads(self.fc_v(value))

        scores = (q @ k.transpose(-1, -2)) / (self.head_dim**0.5)
        global_attn = torch.softmax(scores, dim=-1)
        attn = global_attn
        if adj_mask is not None:
            adj_mask = adj_mask.to(device=scores.device, dtype=torch.bool)
            adj_mask = adj_mask.expand(scores.shape)
            visible = adj_mask.any(dim=-1, keepdim=True)
            # A fully masked row would softmax over all -inf (NaN). Give such rows
            # an all-visible stand-in mask so the softmax stays finite, then zero
            # the whole attention row below. Rows with a visible key are unchanged.
            safe_mask = adj_mask | ~visible
            local_scores = scores.masked_fill(~safe_mask, float("-inf"))
            local_attn = torch.softmax(local_scores, dim=-1)
            attn = torch.where(visible, (global_attn + local_attn) / 2.0, torch.zeros_like(global_attn))

        out = attn @ v  # (..., num_heads, len_q, head_dim)
        out = self._merge_heads(out, batch_shape, len_q)
        return self.out_proj(out)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        *batch_shape, length, _ = x.shape
        x = x.view(*batch_shape, length, self.num_heads, self.head_dim)
        return x.movedim(-2, -3)  # (..., num_heads, length, head_dim)

    def _merge_heads(self, x: torch.Tensor, batch_shape: list[int], length: int) -> torch.Tensor:
        x = x.movedim(-3, -2)  # (..., length, num_heads, head_dim)
        return x.reshape(*batch_shape, length, self.model_dim)


__all__ = ["GlobalLocalGraphAttention"]

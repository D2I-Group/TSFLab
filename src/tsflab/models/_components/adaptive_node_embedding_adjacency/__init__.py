"""Learnable node-embedding adaptive adjacency: ``softmax(relu(E1 @ E2^T))``.

This is the dense, self-normalizing adaptive graph used by several
spatiotemporal forecasters to complement predefined or dynamic supports. It
covers both the dual embedding form (independent source/target parameters)
and the single self-similarity form (one embedding compared against its own
transpose); the pairwise scoring, ``relu``, and row-wise ``softmax`` are
identical in both cases. The node-adaptive Chebyshev filter of AGCRN and HimNet
built on top of it lives in ``node_adaptive_graph_conv``; other polynomial bases
stay model-local.
"""

from __future__ import annotations

import torch


def adaptive_node_embedding_adjacency(source: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``softmax(relu(source @ target), dim=-1)``.

    When ``target`` is omitted, ``source`` is compared against
    ``source.transpose(-1, -2)`` (self-similarity), which also supports
    batched ``(..., nodes, dim)`` embeddings. When ``target`` is given it is
    used exactly as supplied (no implicit transpose), matching call sites
    that already store the target embedding as ``(dim, nodes)``.
    """
    if target is None:
        target = source.transpose(-1, -2)
    return torch.softmax(torch.relu(source @ target), dim=-1)


__all__ = ["adaptive_node_embedding_adjacency"]

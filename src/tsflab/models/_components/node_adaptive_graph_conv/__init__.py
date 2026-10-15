"""Node-adaptive Chebyshev graph convolution (AGCRN's NAPL + DAGG filter).

Each node ``n`` gets its own filter generated from its embedding ``e_n`` through
shared banks, ``W_n = e_n W_bank`` and ``b_n = e_n b_bank``, and the graph is
self-learned from the same embeddings, ``A = softmax(relu(E E^T))``:

``out[b, n] = sum_k (T_k(A) x[b])[n] W_n[k] + b_n``

with ``T_0 = I``, ``T_1 = A`` and ``T_k = 2 A T_{k-1} - T_{k-2}``. The embeddings
may be one static ``[N, D]`` table shared by the batch (AGCRN) or a per-sample
``[B, N, D]`` tensor (HimNet's meta embeddings); both paths keep the exact
operation order of the model code they were extracted from.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)


def _shared_basis(embeddings: torch.Tensor, order: int) -> torch.Tensor:
    """``[N, D] -> [order, N, N]`` stacked Chebyshev basis of the adaptive graph."""
    adjacency = adaptive_node_embedding_adjacency(embeddings)
    basis = [torch.eye(embeddings.shape[0], device=embeddings.device, dtype=embeddings.dtype)]
    if order > 1:
        basis.append(adjacency)
    for _ in range(2, order):
        basis.append(2 * adjacency @ basis[-1] - basis[-2])
    return torch.stack(basis)


class NodeAdaptiveGraphConv(nn.Module):
    """Graph convolution whose per-node weights and bias come from node embeddings.

    Args:
        in_dim: Input feature width ``I``.
        out_dim: Output feature width ``O``.
        order: Number of Chebyshev terms ``K >= 1`` (``T_0 .. T_{K-1}``).
        embed_dim: Width ``D`` of the node embeddings that index the banks.
    """

    def __init__(self, in_dim: int, out_dim: int, order: int, embed_dim: int) -> None:
        super().__init__()
        self.order = order
        self.weight_bank = nn.Parameter(torch.empty(embed_dim, order, in_dim, out_dim))
        self.bias_bank = nn.Parameter(torch.empty(embed_dim, out_dim))
        nn.init.xavier_uniform_(self.weight_bank)
        nn.init.zeros_(self.bias_bank)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Filter ``x [B, N, I]`` with embeddings ``[N, D]`` or ``[B, N, D]``; returns ``[B, N, O]``."""
        if embeddings.dim() == 2:
            basis = _shared_basis(embeddings, self.order)
            neighborhoods = torch.einsum("knm,bmc->bnkc", basis, x)
            weights = torch.einsum("nd,dkio->nkio", embeddings, self.weight_bank)
            bias = embeddings @ self.bias_bank
            return torch.einsum("bnki,nkio->bno", neighborhoods, weights) + bias
        if embeddings.dim() == 3:
            graph = adaptive_node_embedding_adjacency(embeddings)
            identity = torch.eye(embeddings.shape[1], device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1)
            basis = [identity]
            if self.order > 1:
                basis.append(graph)
            for _ in range(2, self.order):
                basis.append(2 * graph @ basis[-1] - basis[-2])
            neighborhoods = torch.einsum("bknm,bmc->bnkc", torch.stack(basis, 1), x)
            weights = torch.einsum("bnd,dkio->bnkio", embeddings, self.weight_bank)
            bias = torch.einsum("bnd,do->bno", embeddings, self.bias_bank)
            return torch.einsum("bnki,bnkio->bno", neighborhoods, weights) + bias
        raise ValueError("embeddings must be [nodes, dim] or [batch, nodes, dim]")


__all__ = ["NodeAdaptiveGraphConv"]

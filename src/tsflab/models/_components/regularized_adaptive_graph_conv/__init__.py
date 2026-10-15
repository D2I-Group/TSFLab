"""Stochastic embedding regularization and a linear-complexity, node-embedding
adaptive graph convolution ("efficient cosine operator").

Two paper-neutral pieces that a node-embedding-based adaptive-graph forecaster
can compose:

``StochasticSharedEmbedding`` regularizes any learnable ``(N, C)`` embedding
table during training by replacing a random ``p``-fraction of *rows* with
another randomly chosen row from the same table (a full-row swap, not a
per-element mask like dropout). This is a general-purpose stochastic
regularizer for *any* embedding matrix (node, entity, or otherwise); it is
returned unchanged in eval mode or when ``p == 0``.

``EfficientCosineGraphConv`` builds a data-driven ("adaptive") graph directly
from a node-embedding table and diffuses a ``(B, N, H)`` hidden state over it
in ``O(N)`` time instead of the ``O(N^2)`` an explicit dense adjacency would
cost. Concretely:

1. **Gated cosine support.** Two learned linear maps turn the node
   embeddings into a softmax "gate" and a ReLU "filter"; their elementwise
   product, L2-normalized per node, is the *support* embedding table
   ``E' in R^{N x C}`` used below in place of a materialized ``N x N``
   adjacency (paper's "Efficient Cosine Operator", cosine similarity of node
   embeddings).
2. **Linearized (kernel-trick) diffusion.** For an implicit adjacency
   ``A = E' E'^T`` (never formed), one hop of degree-normalized propagation
   ``D^{-1} A H`` is computed associatively as
   ``D^{-1} E' (E'^T H)`` — two ``O(N * C * H)`` einsums instead of one
   ``O(N^2 * H)`` matrix multiply, which is the mechanism that keeps this
   adaptive graph convolution linear in the number of nodes ``N`` on
   large-scale road networks. Repeating the hop ``order`` times and
   concatenating every hop's output (including the order-0 input) before one
   shared linear projection mirrors a standard polynomial/Chebyshev-style
   multi-hop graph convolution, but built on the kernelized hop above.

Neither piece is spatiotemporal-specific beyond assuming a leading ``(B, N,
*)`` or ``(N, *)`` node axis, so both compose with any adaptive-graph
forecaster that maintains a learnable per-node embedding table.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticSharedEmbedding(nn.Module):
    """Regularize an embedding table by randomly swapping whole rows.

    Args:
        p: Per-row replacement probability, in ``[0, 1]``. Larger ``p``
            swaps more rows more often.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply stochastic row-swap regularization.

        Args:
            embeddings: Embedding table shaped ``(N, C)``.

        Returns:
            A tensor shaped like ``embeddings``: unchanged outside training
            (or when ``p == 0``), otherwise a copy with a random ``p``
            fraction of rows replaced by another randomly chosen row.
        """
        if not self.training or self.p == 0.0:
            return embeddings
        num_rows = embeddings.shape[0]
        swap_mask = torch.rand(num_rows, device=embeddings.device) < self.p
        random_rows = torch.randint(0, num_rows, (num_rows,), device=embeddings.device)
        swap_mask = swap_mask.unsqueeze(-1)
        return torch.where(swap_mask, embeddings[random_rows], embeddings)


class EfficientCosineGraphConv(nn.Module):
    """Linear-complexity adaptive graph convolution over a node-embedding table.

    Args:
        hidden_dim: Feature width ``H`` of the diffused hidden state.
        spatial_dim: Feature width ``C`` of the node-embedding table.
        order: Number of kernelized diffusion hops ``K >= 1``. Every hop's
            output (plus the order-0 input) is concatenated before the
            shared output projection.
        dropout: Dropout probability applied to the projected output.
            Defaults to ``0.0`` (off); the official reference sets ``0.1``.
    """

    def __init__(self, hidden_dim: int, spatial_dim: int, order: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim < 1 or spatial_dim < 1:
            raise ValueError("hidden_dim and spatial_dim must be positive")
        if order < 1:
            raise ValueError("order must be >= 1")
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.order = order
        self.gate_weight = nn.Parameter(torch.randn(spatial_dim, spatial_dim))
        self.filter_weight = nn.Parameter(torch.randn(spatial_dim, spatial_dim))
        self.out_weight = nn.Parameter(torch.empty(hidden_dim, (order + 1) * hidden_dim))
        nn.init.kaiming_uniform_(self.out_weight, a=5**0.5)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def compute_support(self, node_embedding: torch.Tensor) -> torch.Tensor:
        """Build the gated, L2-normalized support table used by :meth:`forward`.

        Args:
            node_embedding: Node-embedding table shaped ``(N, spatial_dim)``.

        Returns:
            Support table shaped ``(N, spatial_dim)``: an implicit stand-in
            for a dense ``(N, N)`` adjacency, never materialized.
        """
        if node_embedding.shape[-1] != self.spatial_dim:
            raise ValueError(f"expected last axis {self.spatial_dim}, got {node_embedding.shape[-1]}")
        gate = F.softmax(F.relu(node_embedding @ self.gate_weight), dim=1)
        filt = F.relu(node_embedding @ self.filter_weight)
        return F.normalize(gate * filt, p=2, dim=1)

    def _kernelized_hop(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        # D^-1 A x with A = support @ support.T, computed without forming A.
        hidden = torch.einsum("nc,bnh->bhc", support, x)
        hidden = torch.einsum("nc,bhc->bnh", support, hidden)
        degree = torch.einsum("nc,n->c", support, support.new_ones(support.shape[0]))
        degree = torch.einsum("nc,c->n", support, degree)
        degree = degree.unsqueeze(-1) + 1e-6
        return (hidden / degree).contiguous()

    def forward(self, x: torch.Tensor, node_embedding: torch.Tensor) -> torch.Tensor:
        """Diffuse ``x`` over the adaptive graph implied by ``node_embedding``.

        Args:
            x: Hidden state shaped ``(B, N, hidden_dim)``.
            node_embedding: Node-embedding table shaped ``(N, spatial_dim)``.

        Returns:
            Diffused hidden state shaped ``(B, N, hidden_dim)``.
        """
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected last axis {self.hidden_dim}, got {x.shape[-1]}")
        support = self.compute_support(node_embedding)
        hops = [x]
        current = x
        for _ in range(self.order):
            current = self._kernelized_hop(current, support)
            hops.append(current)
        stacked = torch.cat(hops, dim=-1)
        return self.dropout(stacked @ self.out_weight.T)


__all__ = ["StochasticSharedEmbedding", "EfficientCosineGraphConv"]

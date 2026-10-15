"""Shared, input-independent sparse connection routing over discrete positions."""

from __future__ import annotations

import torch
from torch import nn


class SharedSparseConnectionRouter(nn.Module):
    """Learn a shared 0/1 connection matrix over ``num_positions`` positions.

    The matrix is derived from an internal learnable memory table rather than
    from the input batch, so it is shared across every sample and channel that
    consumes it (one matrix per head). At each call the router:

    1. encodes a per-position, per-head memory embedding through a small
       stacked-linear encoder;
    2. scores every ordered pair of positions ``(i, j)`` with a shared
       two-layer MLP over the concatenated pair embeddings;
    3. turns the pair scores into connection probabilities and enforces a
       fixed target density by keeping only the top ``density`` fraction of
       scores per head, using a straight-through estimator so gradients still
       flow to the memory/encoder/scorer parameters through the dense scores.

    During training, standard-Gumbel noise is added to the scores before
    thresholding so the discrete selection explores neighbouring connections
    instead of collapsing deterministically; at evaluation the noise is
    disabled and the router is a deterministic function of its learned
    parameters (it does not depend on ``x`` at all).

    This is a paper-neutral building block for models that mix a fixed number
    of tokens/positions (e.g. patches) through a learned, sparse, sample- and
    channel-shared adjacency instead of a dense (self-attention style) one.
    """

    def __init__(
        self,
        num_positions: int,
        dim: int,
        heads: int = 1,
        density: float = 0.15,
        memory_dim: int | None = None,
        gumbel_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_positions < 1:
            raise ValueError("num_positions must be positive")
        if heads < 1:
            raise ValueError("heads must be positive")
        if not 0.0 < density <= 1.0:
            raise ValueError("density must lie in (0, 1]")
        self.num_positions = num_positions
        self.heads = heads
        self.density = density
        self.gumbel_scale = gumbel_scale
        memory_dim = memory_dim or dim

        self.memory = nn.Embedding(num_positions, memory_dim * heads)
        hidden = max(memory_dim // 2, 1)
        hidden2 = max(memory_dim // 4, 1)
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, hidden),
            nn.Linear(hidden, hidden2),
            nn.Linear(hidden2, memory_dim),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )
        self.register_buffer("_position_index", torch.arange(num_positions), persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(connections, probabilities)``, each ``(heads, N, N)``.

        ``connections`` is binary (straight-through differentiable) and holds
        exactly ``round(density * N * N)`` ones per head. ``probabilities`` is
        the dense sigmoid score used to pick the top connections and is
        returned for inspection or auxiliary regularization by the caller.
        """
        memory = self.memory(self._position_index)
        memory = memory.view(self.num_positions, self.heads, -1).transpose(0, 1)
        memory = self.memory_encoder(memory)  # (heads, N, memory_dim)

        n = self.num_positions
        senders = memory.unsqueeze(2).expand(-1, -1, n, -1)
        receivers = memory.unsqueeze(1).expand(-1, n, -1, -1)
        pairs = torch.cat((senders, receivers), dim=-1)  # (heads, N, N, 2*memory_dim)
        scores = self.pair_scorer(pairs).squeeze(-1)  # (heads, N, N)
        probabilities = torch.sigmoid(scores)

        gated_scores = scores
        if self.training and self.gumbel_scale > 0:
            uniform = torch.rand_like(scores).clamp_min(1e-9)
            gumbel_noise = -torch.log(-torch.log(uniform))
            gated_scores = scores + self.gumbel_scale * gumbel_noise

        k = max(1, round(self.density * n * n))
        flat = gated_scores.reshape(self.heads, -1)
        threshold = flat.kthvalue(flat.shape[-1] - k + 1, dim=-1, keepdim=True).values
        hard = (flat >= threshold).float().reshape(self.heads, n, n)
        soft = torch.sigmoid(gated_scores)
        # Group (soft - soft.detach()) first so it is exactly 0 in floating
        # point before adding to `hard`; `hard + soft - soft.detach()` would
        # instead round `hard + soft` first and not perfectly cancel, leaving
        # `connections` a hair off 0/1 despite carrying `soft`'s gradient.
        connections = hard + (soft - soft.detach())
        return connections, probabilities


__all__ = ["SharedSparseConnectionRouter"]

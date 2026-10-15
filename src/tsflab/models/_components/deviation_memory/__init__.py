"""Learnable prototype memory bank with attention retrieval and a
paper-neutral deviation score between two representations.

Implements a small self-supervised "normal pattern" memory: a bank of ``M``
learnable prototype vectors that any encoder's hidden representation can be
compared against. Given a representation ``h`` of shape ``(..., query_dim)``,
:class:`PrototypeMemory` projects it to a query, attends softly over every
prototype to retrieve a blended value, and additionally reports the single
nearest and second-nearest prototype (by attention score) so a caller can
build a triplet/contrastive objective that pulls representations toward one
prototype and away from a distractor:

    query = h @ Wq                                   # (..., d)
    score = softmax(query @ prototypes^T, dim=-1)     # (..., M)
    value = score @ prototypes                        # (..., d) soft retrieval
    nearest, second_nearest = top-2 prototypes by score

:func:`deviation_score` then reduces any two same-shaped representations
(typically a "current" and a "reference/historical" query or nearest-prototype
pair) to a non-negative scalar per row via the sum of absolute differences over the last axis, giving a
generic self-supervised deviation signal: how different the current pattern's
retrieval is from a reference one. Neither piece is spatiotemporal-specific —
``h`` may carry a node axis, a plain batch axis, or any other leading shape —
so this composes with graph forecasters and ordinary sequence forecasters
alike that want an auxiliary "how different from the past" signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PrototypeRetrieval:
    """Outputs of one :class:`PrototypeMemory` query.

    Attributes:
        value: Attention-weighted combination of every prototype, shaped
            ``(..., prototype_dim)``. The soft retrieval used to augment a
            downstream decoder.
        query: The projected query used to score prototypes, shaped
            ``(..., prototype_dim)``.
        nearest: The single closest prototype by attention score, shaped
            ``(..., prototype_dim)``.
        second_nearest: The second-closest prototype by attention score,
            shaped ``(..., prototype_dim)``.
        indices: Indices of ``(nearest, second_nearest)`` in the prototype
            bank, shaped ``(..., 2)``.
    """

    value: torch.Tensor
    query: torch.Tensor
    nearest: torch.Tensor
    second_nearest: torch.Tensor
    indices: torch.Tensor


class PrototypeMemory(nn.Module):
    """Learnable prototype bank with soft attention retrieval and top-2 lookup.

    Args:
        query_dim: Feature width of the input representation ``h``.
        prototype_dim: Feature width of each prototype vector and of the
            projected query.
        num_prototypes: Number of learnable prototype vectors ``M``.
    """

    def __init__(self, query_dim: int, prototype_dim: int, num_prototypes: int) -> None:
        super().__init__()
        if min(query_dim, prototype_dim, num_prototypes) < 1:
            raise ValueError("PrototypeMemory dimensions must be positive")
        if num_prototypes < 2:
            raise ValueError("PrototypeMemory requires at least 2 prototypes for top-2 lookup")
        self.query_dim = query_dim
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        self.query_proj = nn.Parameter(torch.empty(query_dim, prototype_dim))
        nn.init.xavier_normal_(self.prototypes)
        nn.init.xavier_normal_(self.query_proj)

    def forward(self, h: torch.Tensor) -> PrototypeRetrieval:
        """Query the prototype bank with representation ``h``.

        Args:
            h: Representation tensor shaped ``(..., query_dim)``.

        Returns:
            A :class:`PrototypeRetrieval` whose fields share ``h``'s leading
            shape with the last axis replaced by ``prototype_dim`` (or ``2``
            for ``indices``).
        """
        if h.shape[-1] != self.query_dim:
            raise ValueError(f"expected last axis {self.query_dim}, got {h.shape[-1]}")
        query = h @ self.query_proj  # (..., d)
        scores = torch.softmax(query @ self.prototypes.t(), dim=-1)  # (..., M)
        value = scores @ self.prototypes  # (..., d)
        top_scores, indices = torch.topk(scores, k=2, dim=-1)
        del top_scores
        nearest = self.prototypes[indices[..., 0]]
        second_nearest = self.prototypes[indices[..., 1]]
        return PrototypeRetrieval(
            value=value, query=query, nearest=nearest, second_nearest=second_nearest, indices=indices
        )


def deviation_score(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Summed absolute deviation between two same-shaped representations.

    Computes ``|current - reference|.sum(-1)``: an L1 distance, not a mean, so
    the value scales with ``D``.

    Args:
        current: Tensor shaped ``(..., D)``.
        reference: Tensor shaped ``(..., D)``, broadcastable to ``current``.

    Returns:
        Non-negative tensor shaped ``(...,)``: how different ``current`` is
        from ``reference`` along the last axis.
    """
    if current.shape[-1] != reference.shape[-1]:
        raise ValueError("current and reference must share their last axis width")
    return torch.abs(current - reference).sum(dim=-1)


__all__ = ["PrototypeMemory", "PrototypeRetrieval", "deviation_score"]

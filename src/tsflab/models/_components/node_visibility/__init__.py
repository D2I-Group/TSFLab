"""Node-visibility masking and subgraph grouping for scalable node-set attention.

These helpers implement a reusable "node visibility" mechanism: a random
node-level keep/drop mask that shrinks the visible token set before an
expensive all-pairs operation (e.g. self-attention over nodes), plus a
shuffle/group/ungroup/unshuffle pipeline that partitions the (already
shrunk) token set into fixed-size subgraphs so attention cost scales with
the subgraph size instead of the full node count. All functions operate on
a generic ``(B, L, D)`` token tensor along the token axis ``L`` and are
agnostic to what ``L`` indexes (nodes, patches, or any other token kind),
so they compose with any node/token-structured encoder.
"""

from __future__ import annotations

import torch


def random_mask_tokens(
    x: torch.Tensor, mask_ratio: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly keep a ``(1 - mask_ratio)`` fraction of tokens along dim 1.

    Args:
        x: Token tensor shaped ``(B, L, D)``.
        mask_ratio: Fraction of tokens to drop, in ``[0.0, 1.0)``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        A tuple ``(kept, keep_indices)`` where ``kept`` is shaped
        ``(B, keep_length, D)`` and ``keep_indices`` is a 1-D tensor of the
        sorted token positions kept (shared across the batch), shaped
        ``(keep_length,)``. When ``mask_ratio <= 0`` the input is returned
        unchanged together with the full index range.
    """
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in [0.0, 1.0)")
    _, length, _ = x.shape
    if mask_ratio <= 0.0:
        return x, torch.arange(length, device=x.device)
    keep_length = max(1, int(length * (1 - mask_ratio)))
    permutation = torch.randperm(length, device=x.device, generator=generator)
    keep_indices, _ = torch.sort(permutation[:keep_length])
    kept = torch.gather(x, 1, keep_indices.view(1, -1, 1).expand(x.shape[0], -1, x.shape[-1]))
    return kept, keep_indices


def shuffle_tokens(x: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent random per-sample permutation along dim 1.

    Args:
        x: Token tensor shaped ``(B, L, D)``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        A tuple ``(shuffled, perm)`` where ``perm`` is shaped ``(B, L)`` and
        records, per batch row, the token index placed at each output
        position; pass it to :func:`unshuffle_tokens` to invert.
    """
    _, length, _ = x.shape
    scores = torch.rand(x.shape[0], length, device=x.device, generator=generator)
    perm = torch.argsort(scores, dim=1)
    shuffled = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return shuffled, perm


def unshuffle_tokens(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Invert :func:`shuffle_tokens` given the permutation it returned."""
    inverse = torch.argsort(perm, dim=1)
    return torch.gather(x, 1, inverse.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def group_into_subgraphs(x: torch.Tensor, subgraph_size: int) -> tuple[torch.Tensor, int, int]:
    """Partition tokens into zero-padded fixed-size subgraph groups.

    Args:
        x: Token tensor shaped ``(B, L, D)``.
        subgraph_size: Target group size along the token axis. Must be
            positive.

    Returns:
        A tuple ``(grouped, num_groups, orig_length)``. When ``L <=
        subgraph_size`` the input is returned unchanged with
        ``num_groups=1``. Otherwise ``grouped`` is shaped
        ``(B * num_groups, subgraph_size, D)``, right zero-padded along the
        token axis so ``num_groups * subgraph_size >= L``. Pass all three
        return values to :func:`ungroup_subgraphs` to invert.
    """
    if subgraph_size <= 0:
        raise ValueError("subgraph_size must be positive")
    batch_size, length, width = x.shape
    if length <= subgraph_size:
        return x, 1, length
    num_groups = (length + subgraph_size - 1) // subgraph_size
    padded_length = num_groups * subgraph_size
    if length < padded_length:
        pad = torch.zeros(batch_size, padded_length - length, width, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    grouped = x.reshape(batch_size * num_groups, subgraph_size, width)
    return grouped, num_groups, length


def ungroup_subgraphs(x: torch.Tensor, num_groups: int, orig_length: int, subgraph_size: int) -> torch.Tensor:
    """Invert :func:`group_into_subgraphs`, dropping any padding tokens."""
    if orig_length <= subgraph_size:
        return x
    grouped_size = x.shape[1]
    batch_size = x.shape[0] // num_groups
    width = x.shape[-1]
    merged = x.reshape(batch_size, num_groups * grouped_size, width)
    return merged[:, :orig_length, :]


__all__ = [
    "random_mask_tokens",
    "shuffle_tokens",
    "unshuffle_tokens",
    "group_into_subgraphs",
    "ungroup_subgraphs",
]

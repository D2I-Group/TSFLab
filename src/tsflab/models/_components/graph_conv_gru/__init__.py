"""GRU recurrence whose gate and candidate maps are graph convolutions.

The convolutional-recurrent cell of GCRN and DCRNN replaces the two affine maps
of a GRU with graph filters ``G_g`` (width ``2H``) and ``G_c`` (width ``H``):

``r, u = split(sigmoid(G_g([x, h])))``,
``c = tanh(G_c([x, r * h]))``,
``h' = u * h + (1 - u) * c``.

Only this gating is shared. The graph filter (diffusion, Chebyshev,
node-adaptive, dynamic, ...) and its graph arguments stay with the caller and
are passed in as callables or modules.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def graph_gru_step(
    x: torch.Tensor,
    state: torch.Tensor,
    gates: Callable[[torch.Tensor], torch.Tensor],
    candidate: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """One graph-GRU update ``[B, N, I], [B, N, H] -> [B, N, H]``.

    ``gates`` maps the joined ``[x, h]`` (``[B, N, I + H]``) to ``[B, N, 2H]``
    pre-activations whose first half is the reset gate and second half the update
    gate; ``candidate`` maps ``[x, r * h]`` to ``[B, N, H]``.
    """
    reset, update = torch.sigmoid(gates(torch.cat((x, state), dim=-1))).chunk(2, dim=-1)
    proposal = torch.tanh(candidate(torch.cat((x, reset * state), dim=-1)))
    return update * state + (1.0 - update) * proposal


class GraphConvGRUCell(nn.Module):
    """Graph-GRU cell around two caller-built graph-convolution modules.

    ``gates`` and ``candidate`` are registered as submodules under those names (so
    their parameters appear as ``gates.*`` and ``candidate.*``) and are called as
    ``module(joined, *graph)``, where ``graph`` is whatever extra positional
    arguments ``forward`` receives after the state.
    """

    def __init__(self, gates: nn.Module, candidate: nn.Module) -> None:
        super().__init__()
        self.gates = gates
        self.candidate = candidate

    def forward(self, x: torch.Tensor, state: torch.Tensor, *graph: object) -> torch.Tensor:
        """Return the next hidden state ``[B, N, H]``."""
        return graph_gru_step(
            x,
            state,
            lambda joined: self.gates(joined, *graph),
            lambda joined: self.candidate(joined, *graph),
        )


__all__ = ["GraphConvGRUCell", "graph_gru_step"]

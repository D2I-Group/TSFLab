"""Local VisiFold implementation from the paper and pinned reference details.

VisiFold (arXiv:2603.11816, ICDE 2026) targets long-term traffic forecasting
with two ideas:

* A *temporal folding graph*: instead of stacking ``T`` temporal snapshots of
  an ``N``-node graph and message-passing across both axes, the whole input
  history for every node is folded (flattened) into a single per-node token
  through a small MLP. The result is one graph with ``N`` tokens, each
  carrying the entire folded history, so all node-mixing happens through a
  single self-attention pass over the (already-folded) node axis instead of
  ``T`` repeated spatial passes.
* A *node visibility* mechanism used only during training: node-level random
  masking drops a ``mask_ratio`` fraction of node tokens, the remaining
  tokens are shuffled, and grouped into fixed-size subgraphs so self-attention
  cost scales with the subgraph size rather than the full (visible) node
  count; the transform is inverted after attention so the loss can still be
  computed against every masked-in node's target.

Both the fold (a strided ``Linear`` over the time axis) and the visibility
pipeline (mask/shuffle/group + inverse) mirror the official
``model/VisiFold.py`` at the pinned revision. The folding MLP is kept
model-local (it is a two-layer per-node projection, not a distinct reusable
contract); the node-visibility mask/shuffle/group/ungroup pipeline is
reusable across node-structured attention models and is imported from
``tsflab.models._components.node_visibility``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._components.node_visibility import (
    group_into_subgraphs,
    random_mask_tokens,
    shuffle_tokens,
    ungroup_subgraphs,
    unshuffle_tokens,
)


class SelfAttentionLayer(nn.Module):
    """Post-norm self-attention block over the (folded) node-token axis.

    Matches the official ``SelfAttentionLayer``: attention residual is added
    then layer-normalized, followed by a GELU feed-forward residual that is
    added then layer-normalized. The official ``MultiHeadAttention`` supports
    an optional additive edge-embedding bias, but VisiFold's own ``forward``
    never passes one, so this local block uses ``nn.MultiheadAttention``
    directly (a faithful simplification recorded in the model card).
    """

    def __init__(self, model_dim: int, feed_forward_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(attended))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


class Model(nn.Module):
    """Temporal-folding graph with node-visibility masked self-attention."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        steps_per_day: int = 24,
        input_embedding_dim: int = 8,
        tod_embedding_dim: int = 4,
        dow_embedding_dim: int = 4,
        spatial_embedding_dim: int = 8,
        feed_forward_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        mask_ratio: float = 0.2,
        subgraph_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del adj_mx  # VisiFold's node visibility mechanism does not consume a fixed adjacency.
        if min(seq_len, pred_len, num_nodes) < 1:
            raise ValueError("VisiFold dimensions must be positive")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.mask_ratio = mask_ratio
        self.subgraph_size = subgraph_size

        self.model_dim = input_embedding_dim + tod_embedding_dim + dow_embedding_dim + spatial_embedding_dim

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # Temporal folding: collapse the whole (in_steps) history of each node
        # into a single token, rather than keeping a per-timestep node axis.
        self.fold_input = nn.Sequential(
            nn.Linear(seq_len, input_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_embedding_dim, input_embedding_dim),
        )
        self.attn_layers = nn.ModuleList(
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)
        )
        self.output_head = nn.Sequential(
            nn.Linear(self.model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, pred_len),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"VisiFold expects (B, {self.seq_len}, {self.num_nodes}) values")
        batch_size = x_enc.shape[0]
        history = to_spatiotemporal(x_enc, x_mark_enc)

        # Temporal folding graph: (B, T, N) -> (B, N, T) -> (B, N, D_in).
        tokens = self.fold_input(history[..., 0].transpose(1, 2))
        pieces = [tokens]
        if self.tod_embedding_dim > 0:
            index = (history[:, -1, :, 1] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            pieces.append(self.tod_embedding(index))
        if self.dow_embedding_dim > 0:
            index = (history[:, -1, :, 2] * 7).long().clamp(0, 6)
            pieces.append(self.dow_embedding(index))
        if self.spatial_embedding_dim > 0:
            pieces.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1))
        tokens = torch.cat(pieces, dim=-1)  # (B, N, model_dim)

        num_groups = 1
        keep_indices: torch.Tensor | None = None
        self.last_keep_indices = None
        shuffle_perm: torch.Tensor | None = None
        if self.training:
            # Node visibility: mask a fraction of nodes, shuffle the survivors,
            # then split them into fixed-size subgraphs before attention.
            tokens, keep_indices = random_mask_tokens(tokens, self.mask_ratio)
            self.last_keep_indices = keep_indices.detach()
            tokens, shuffle_perm = shuffle_tokens(tokens)
            visible_length = tokens.shape[1]
            tokens, num_groups, visible_length = group_into_subgraphs(tokens, self.subgraph_size)

        for layer in self.attn_layers:
            tokens = layer(tokens)

        tokens = self.output_head(tokens)  # (B or B*num_groups, N_visible|subgraph, pred_len)

        if self.training:
            tokens = ungroup_subgraphs(tokens, num_groups, visible_length, self.subgraph_size)
            tokens = unshuffle_tokens(tokens, shuffle_perm)
            # Scatter predictions for masked-out nodes back to zero so the
            # output keeps the full (B, N, pred_len) contract; the training
            # loss should be restricted to `keep_indices` for those samples.
            full = tokens.new_zeros(batch_size, self.num_nodes, self.pred_len)
            index = keep_indices.view(1, -1, 1).expand(batch_size, -1, self.pred_len)
            full.scatter_(1, index, tokens)
            tokens = full

        return tokens.transpose(1, 2).contiguous()  # (B, pred_len, N)


__all__ = ["Model", "SelfAttentionLayer"]

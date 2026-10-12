"""TwinFormer: dual-level patch Transformer with GRU aggregation (Kumavat and Maheshwari, 2025).

The multivariate window is embedded per time step (Eq. 6) and split into
non-overlapping patches (Eq. 7). A shared Local Informer block applies top-k
sparse attention and a feed-forward network inside every patch (Eqs. 8-9), the
patch is mean-pooled to one token (Eq. 10), a Global Informer block attends
across patch tokens (Eqs. 12-13), a GRU reads the patch tokens in order
(Eqs. 14-17), and a linear head maps its final hidden state to the horizon (Eq. 18).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import PositionalEmbedding


def topk_sparse_softmax(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Eqs. (2)-(3): keep the ``k`` largest logits of each row, ``-inf`` elsewhere, softmax.

    Exactly ``min(k, n)`` entries per row stay (ties are broken by ``torch.topk``).
    """
    count = min(k, logits.shape[-1])
    values, index = logits.topk(count, dim=-1)
    masked = torch.full_like(logits, float("-inf")).scatter(-1, index, values)
    return torch.softmax(masked, dim=-1)


class TopKSparseAttention(nn.Module):
    """Multi-head custom sparse attention (Eqs. 1-4) with biased Q/K/V/output projections."""

    def __init__(self, d_model: int, n_heads: int, top_k: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.top_k = top_k
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        head = width // self.n_heads

        def split(layer: nn.Linear) -> torch.Tensor:
            return layer(x).view(batch, length, self.n_heads, head).transpose(1, 2)

        logits = split(self.query) @ split(self.key).transpose(-1, -2) / math.sqrt(head)  # Eq. (1)
        context = topk_sparse_softmax(logits, self.top_k) @ split(self.value)  # Eqs. (2)-(4)
        return self.out(context.transpose(1, 2).reshape(batch, length, width))


class InformerBlock(nn.Module):
    """Eqs. (8)-(9) / (12)-(13): ``Z1 = Z + CSA(Z)``, ``Z2 = Z1 + FFN(LayerNorm(Z1))``."""

    def __init__(self, d_model: int, n_heads: int, top_k: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = TopKSparseAttention(d_model, n_heads, top_k)
        self.norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x))
        hidden = self.dropout(F.gelu(self.ff1(self.norm(x))))
        return x + self.dropout(self.ff2(hidden))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        patch_len: int = 12,
        top_k: int = 5,
        local_layers: int = 1,
        global_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len  # Np = floor(L / P)
        self.embedding = nn.Linear(enc_in, d_model)  # Eq. (6)
        self.position = PositionalEmbedding(d_model, max_len=seq_len)
        self.position_dropout = nn.Dropout(dropout)
        self.local_blocks = nn.ModuleList(
            InformerBlock(d_model, n_heads, top_k, d_ff, dropout) for _ in range(local_layers)
        )
        self.global_blocks = nn.ModuleList(
            InformerBlock(d_model, n_heads, top_k, d_ff, dropout) for _ in range(global_layers)
        )
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, pred_len * enc_in)  # Eq. (18), one block per channel

    def patch_tokens(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``[B, L, C] -> [B, Np, d]``: embed, keep the newest ``Np * P`` steps, local blocks, mean pool."""
        batch = x_enc.shape[0]
        tokens = self.embedding(x_enc)
        tokens = self.position_dropout(tokens + self.position(tokens))
        tokens = tokens[:, x_enc.shape[1] - self.num_patches * self.patch_len:]
        patches = tokens.reshape(batch * self.num_patches, self.patch_len, -1)  # Eq. (7)
        for block in self.local_blocks:
            patches = block(patches)
        return patches.mean(dim=1).view(batch, self.num_patches, -1)  # Eq. (10)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        tokens = self.patch_tokens(x_enc)
        for block in self.global_blocks:
            tokens = block(tokens)
        _, hidden = self.gru(tokens)  # Eqs. (14)-(17), h_0 = 0
        forecast = self.head(hidden[-1])
        return forecast.view(x_enc.shape[0], self.pred_len, self.enc_in)

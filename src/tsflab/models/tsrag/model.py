"""Clean-room TS-RAG retrieval and Adaptive Retrieval Mixer implementation.

The module implements paper Equations (1)--(12): a context/future knowledge
base, Euclidean top-k retrieval, future projection, attention/FFN interaction,
adaptive softmax mixing, a query skip, and forecast projection. It accepts an
external knowledge base and provides a deterministic history-derived fallback
for the repository's standalone forecasting contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class AdaptiveRetrievalMixer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        # A shared scalar bias would cancel exactly inside the item-wise softmax.
        self.scoring = nn.Linear(d_model, 1, bias=False)

    def forward(self, query: torch.Tensor, retrieved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([query.unsqueeze(1), retrieved], dim=1)
        attended, _ = self.attention(combined, combined, combined, need_weights=False)
        attended = self.attention_norm(combined + attended)
        contextual = self.feedforward_norm(attended + self.feedforward(attended))
        weights = torch.softmax(self.scoring(contextual).squeeze(-1), dim=-1)
        mixed = query + torch.sum(weights.unsqueeze(-1) * contextual, dim=1)
        return mixed, weights


def _per_sample(knowledge: torch.Tensor, batch: int) -> torch.Tensor:
    """Broadcast a shared ``(N, ...)`` knowledge base to ``(B, N, ...)``."""
    if knowledge.ndim == 3:
        return knowledge.unsqueeze(0).expand(batch, -1, -1, -1)
    return knowledge


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        top_k: int = 4,
        memory_size: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.top_k = top_k
        self.memory_size = memory_size
        self.d_model = d_model
        self.revin = RevIN(enc_in)
        self.query_backbone = nn.Sequential(
            nn.Linear(seq_len * enc_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.retrieved_projector = nn.Sequential(
            nn.Linear(pred_len * enc_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.arm = AdaptiveRetrievalMixer(d_model, num_heads, dropout)
        self.output_projection = nn.Linear(d_model, pred_len * enc_in)

    def retrieval_embedding(self, contexts: torch.Tensor) -> torch.Tensor:
        """Create a deterministic compact retrieval descriptor."""
        flattened = contexts.flatten(-2)
        leading = flattened.shape[:-1]
        pooled = F.adaptive_avg_pool1d(flattened.reshape(-1, 1, flattened.shape[-1]), self.d_model).squeeze(1)
        return pooled.reshape(*leading, self.d_model)

    def build_local_knowledge(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the documented standalone fallback knowledge pairs."""
        contexts = []
        futures = []
        for shift in range(1, self.memory_size + 1):
            candidate = torch.roll(normalized, shifts=shift, dims=1)
            contexts.append(candidate)
            futures.append(
                F.interpolate(
                    candidate.transpose(1, 2),
                    self.pred_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            )
        return torch.stack(contexts, dim=1), torch.stack(futures, dim=1)

    def retrieve(self, query: torch.Tensor, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k candidate indices and Euclidean distances (Eq. 2--4)."""
        query_embedding = self.retrieval_embedding(query).unsqueeze(1)
        context_embedding = self.retrieval_embedding(contexts)
        distances = torch.linalg.vector_norm(context_embedding - query_embedding, dim=-1)
        selected_distances, indices = torch.topk(distances, k=min(self.top_k, contexts.shape[1]), largest=False)
        return indices, selected_distances

    def _knowledge(
        self,
        normalized: torch.Tensor,
        contexts: torch.Tensor | None,
        futures: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if contexts is None and futures is None:
            return self.build_local_knowledge(normalized)
        if contexts is None or futures is None:
            raise ValueError("retrieval_contexts and retrieval_futures must be supplied together")
        batch = normalized.shape[0]
        contexts, futures = _per_sample(contexts, batch), _per_sample(futures, batch)
        self._check_knowledge(contexts, futures, batch)
        return contexts.to(normalized), futures.to(normalized)

    def _check_knowledge(self, contexts: torch.Tensor, futures: torch.Tensor, batch: int) -> None:
        if contexts.ndim != 4 or contexts.shape[0] != batch:
            raise ValueError("retrieval_contexts must have shape (N,L,C) or (B,N,L,C)")
        if contexts.shape[2:] != (self.seq_len, self.enc_in):
            raise ValueError("retrieval context shape does not match the model contract")
        if futures.shape != (batch, contexts.shape[1], self.pred_len, self.enc_in):
            raise ValueError("retrieval future shape does not match the model contract")

    def forecast_with_retrieval(
        self,
        x_enc: torch.Tensor,
        *,
        retrieval_contexts: torch.Tensor | None = None,
        retrieval_futures: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        contexts, futures = self._knowledge(normalized, retrieval_contexts, retrieval_futures)
        indices, _ = self.retrieve(normalized, contexts)
        gather_index = indices[:, :, None, None].expand(-1, -1, self.pred_len, self.enc_in)
        retrieved_futures = torch.gather(futures, 1, gather_index)
        retrieved = self.retrieved_projector(retrieved_futures.flatten(-2))
        query = self.query_backbone(normalized.flatten(1))
        mixed, _ = self.arm(query, retrieved)
        forecast = self.output_projection(mixed).reshape(x_enc.shape[0], self.pred_len, self.enc_in)
        return self.revin(forecast, "denorm")

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        return self.forecast_with_retrieval(x_enc)


__all__ = ["AdaptiveRetrievalMixer", "Model"]

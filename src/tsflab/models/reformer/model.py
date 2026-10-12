"""Clean-room Reformer with sparse LSH attention and reversible coupling."""

from __future__ import annotations

import math

import torch
from torch import nn


class LSHSelfAttention(nn.Module):
    """Shared-QK attention after random-projection hashing, sorting, and chunking.

    The implementation never materializes an ``L x L`` tensor. Each sorted chunk
    attends to itself and one chunk back, with original-position causal masking
    when requested. Duplicate candidates across hash rounds receive the paper's
    logarithmic multiplicity correction.
    """

    def __init__(
        self,
        dim,
        heads=4,
        bucket_size=8,
        n_hashes=2,
        dropout=0.0,
        causal=False,
        max_sequence_length=512,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        if min(bucket_size, n_hashes, max_sequence_length) < 1:
            raise ValueError(
                "bucket_size, n_hashes, and maximum length must be positive"
            )
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.bucket_size, self.n_hashes, self.causal = bucket_size, n_hashes, causal
        max_buckets = max(2, 2 * math.ceil(max_sequence_length / (2 * bucket_size)))
        generator = torch.Generator().manual_seed(20010405 + dim + heads + n_hashes)
        rotations = torch.randn(
            n_hashes, self.head_dim, max_buckets // 2, generator=generator
        )
        self.register_buffer("rotations", rotations)
        self.to_qk = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.last_candidate_width = 0

    def hash_vectors(self, vectors):
        length = vectors.shape[1]
        n_buckets = max(2, 2 * math.ceil(length / (2 * self.bucket_size)))
        half = n_buckets // 2
        if half > self.rotations.shape[-1]:
            raise ValueError("sequence exceeds configured max_sequence_length")
        rotated = torch.einsum("bld,rde->rble", vectors, self.rotations[..., :half])
        return torch.cat((rotated, -rotated), dim=-1).argmax(dim=-1)

    def _attend_bucket(self, state, sample, members):
        """Attend each chunk of one bucket's sorted ``members`` to itself and the chunk before."""
        for start in range(0, members.numel(), self.bucket_size):
            query_indices = members[start : start + self.bucket_size]
            candidate_start = max(0, start - self.bucket_size)
            key_indices = members[candidate_start : start + self.bucket_size]
            self._attend_chunk(state, sample, query_indices, key_indices)

    def _attend_chunk(self, state, sample, query_indices, key_indices):
        """Accumulate one chunk's attention output and normalizer into ``state``."""
        qk, projected_values, buckets, accumulated, normalizer = state
        self.last_candidate_width = max(
            self.last_candidate_width, key_indices.numel()
        )
        scores = qk[sample, query_indices] @ qk[
            sample, key_indices
        ].transpose(0, 1)
        scores = scores / math.sqrt(self.head_dim)
        allowed = torch.ones_like(scores, dtype=torch.bool)
        if self.causal:
            allowed = key_indices[None, :] <= query_indices[:, None]
        # Count repeated query-key collisions over all hash rounds.
        duplicate_count = (
            (
                buckets[:, sample, query_indices, None]
                == buckets[:, sample, key_indices][..., None, :]
            )
            .sum(dim=0)
            .clamp_min(1)
        )
        scores = scores - duplicate_count.log()
        scores = scores.masked_fill(~allowed, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        update = (
            self.dropout(weights)
            @ projected_values[sample, key_indices]
        )
        accumulated[sample].index_add_(0, query_indices, update)
        normalizer[sample].index_add_(
            0,
            query_indices,
            allowed.any(dim=-1, keepdim=True).to(normalizer.dtype),
        )

    def forward(self, values, qk_input=None):
        if values.ndim != 3:
            raise ValueError("LSH attention input must be [batch, length, dim]")
        qk_input = values if qk_input is None else qk_input
        batch, length, _ = values.shape
        heads, width = self.heads, self.head_dim
        qk = (
            self.to_qk(qk_input)
            .view(batch, length, heads, width)
            .transpose(1, 2)
            .reshape(batch * heads, length, width)
        )
        qk = qk / qk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        projected_values = (
            self.to_v(values)
            .view(batch, length, heads, width)
            .transpose(1, 2)
            .reshape(batch * heads, length, width)
        )
        buckets = self.hash_vectors(qk.detach())
        accumulated = torch.zeros_like(projected_values)
        normalizer = torch.zeros(
            batch * heads, length, 1, device=values.device, dtype=values.dtype
        )
        positions = torch.arange(length, device=values.device)
        self.last_candidate_width = 0

        state = (qk, projected_values, buckets, accumulated, normalizer)
        for round_index in range(self.n_hashes):
            for sample in range(batch * heads):
                bucket = buckets[round_index, sample]
                order = torch.argsort(bucket * length + positions, stable=True)
                # Chunk each bucket separately. This is equivalent to sorting by
                # (bucket, position), but prevents future-only bucket population
                # from shifting a past bucket's chunk boundaries in causal mode.
                for bucket_index in bucket.unique(sorted=True):
                    self._attend_bucket(state, sample, order[bucket[order] == bucket_index])
        output = accumulated / normalizer.clamp_min(1.0)
        output = (
            output.view(batch, heads, length, width)
            .transpose(1, 2)
            .reshape(batch, length, self.dim)
        )
        return self.to_out(output)


class ReversibleBlock(nn.Module):
    """Algebraic reversible residual block: y1=x1+F(x2), y2=x2+G(y1)."""

    def __init__(
        self,
        half_dim,
        n_heads,
        d_ff,
        bucket_size,
        n_hashes,
        dropout,
        causal,
        max_sequence_length,
    ):
        super().__init__()
        self.f = LSHSelfAttention(
            half_dim,
            n_heads,
            bucket_size,
            n_hashes,
            dropout,
            causal,
            max_sequence_length,
        )
        self.g = nn.Sequential(
            nn.LayerNorm(half_dim),
            nn.Linear(half_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, half_dim),
        )

    def forward(self, hidden):
        first, second = hidden.chunk(2, dim=-1)
        y1 = first + self.f(second)
        y2 = second + self.g(y1)
        return torch.cat((y1, y2), dim=-1)

    def inverse(self, output):
        y1, y2 = output.chunk(2, dim=-1)
        second = y2 - self.g(y1)
        first = y1 - self.f(second)
        return torch.cat((first, second), dim=-1)


class TimeValueEmbedding(nn.Module):
    def __init__(self, channels, d_model, dropout):
        super().__init__()
        self.values = nn.Linear(channels, d_model)
        self.calendar = nn.Linear(6, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, marks):
        hidden = self.values(values)
        if marks is not None:
            if marks.shape[:2] != values.shape[:2] or marks.shape[-1] != 6:
                raise ValueError("marks must have shape [batch, length, 6]")
            hidden = hidden + self.calendar(marks.to(values.dtype))
        return self.dropout(hidden)


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out=None,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        activation="gelu",
        bucket_size=4,
        n_hashes=4,
        causal=False,
    ):
        super().__init__()
        c_out = enc_in if c_out is None else c_out
        total_length = seq_len + pred_len
        if (
            min(
                seq_len,
                pred_len,
                enc_in,
                c_out,
                d_model,
                n_heads,
                e_layers,
                d_ff,
                bucket_size,
                n_hashes,
            )
            < 1
        ):
            raise ValueError("all Reformer dimensions and counts must be positive")
        if d_model % (2 * n_heads):
            raise ValueError("d_model/2 must be divisible by n_heads")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.embedding = TimeValueEmbedding(enc_in, d_model, dropout)
        self.layers = nn.ModuleList(
            [
                ReversibleBlock(
                    d_model // 2,
                    n_heads,
                    d_ff,
                    bucket_size,
                    n_hashes,
                    dropout,
                    causal,
                    total_length,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, c_out)

    def _future(self, x_enc, x_dec):
        """Decoder placeholders for the horizon: zeros, or the last ``pred_len`` steps of ``x_dec``."""
        if x_dec is None:
            return x_enc.new_zeros(x_enc.shape[0], self.pred_len, self.enc_in)
        if (
            x_dec.shape[0] != x_enc.shape[0]
            or x_dec.shape[-1] != self.enc_in
            or x_dec.shape[1] < self.pred_len
        ):
            raise ValueError(
                "x_dec must supply at least pred_len future placeholders"
            )
        return x_dec[:, -self.pred_len :]

    def _marks(self, x_mark_enc, x_mark_dec):
        """History and horizon marks concatenated, or ``None`` when neither is given."""
        if x_mark_enc is None and x_mark_dec is None:
            return None
        if (
            x_mark_enc is None
            or x_mark_dec is None
            or x_mark_dec.shape[1] < self.pred_len
        ):
            raise ValueError("encoder and decoder marks must be supplied together")
        return torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len :]), dim=1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]"
            )
        values = torch.cat((x_enc, self._future(x_enc, x_dec)), dim=1)
        hidden = self.embedding(values, self._marks(x_mark_enc, x_mark_dec))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output_projection(self.norm(hidden[:, -self.pred_len :]))

"""Compact clean-room Kronos: hierarchical BSQ tokens and causal generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


def _binary_codebook(bits: int) -> torch.Tensor:
    indices = torch.arange(2**bits, dtype=torch.long).unsqueeze(1)
    shifts = torch.arange(bits - 1, -1, -1, dtype=torch.long).unsqueeze(0)
    return ((indices >> shifts) & 1).float().mul_(2).sub_(1)


class CausalBlock(nn.Module):
    """Decoder-only Transformer block with an incremental attention cache."""

    def __init__(self, dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.attention_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query = self._heads(self.query(x))
        key = self._heads(self.key(x))
        value = self._heads(self.value(x))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attended = torch.softmax(scores, -1) @ value
        attended = attended.transpose(1, 2).reshape_as(x)
        x = self.attention_norm(x + self.dropout(self.output(attended)))
        x = self.feed_forward_norm(x + self.dropout(self.feed_forward(x)))
        return x, (key, value)

    def step(
        self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        old_key, old_value = cache
        query = self._heads(self.query(x))
        key = torch.cat((old_key, self._heads(self.key(x))), dim=2)
        value = torch.cat((old_value, self._heads(self.value(x))), dim=2)
        weights = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.head_dim), -1)
        attended = (weights @ value).transpose(1, 2).reshape_as(x)
        x = self.attention_norm(x + self.dropout(self.output(attended)))
        x = self.feed_forward_norm(x + self.dropout(self.feed_forward(x)))
        return x, (key, value)


class HierarchicalTokenizer(nn.Module):
    """BSQ tokenizer with an explicitly factorized coarse/fine binary code."""

    def __init__(self, channels: int, dimension: int, code_bits: int) -> None:
        super().__init__()
        self.channels = channels
        self.code_bits = code_bits
        self.encoder = nn.Linear(channels, dimension)
        self.bit_projection = nn.Linear(dimension, code_bits)
        self.decoder = nn.Linear(code_bits, channels)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.bit_projection(F.gelu(self.encoder(x)))
        spherical = F.normalize(latent, dim=-1) * math.sqrt(self.code_bits)
        hard = torch.where(spherical >= 0, torch.ones_like(spherical), -torch.ones_like(spherical))
        straight_through = spherical + (hard - spherical).detach()
        return straight_through, spherical

    def hierarchical_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        bits, continuous = self.encode(x)
        half = self.code_bits // 2
        coarse = torch.cat((bits[..., :half], torch.zeros_like(bits[..., half:])), -1)
        coarse_loss = F.mse_loss(self.decoder(coarse), x)
        fine_loss = F.mse_loss(self.decoder(bits), x)
        commitment = F.mse_loss(continuous, bits.detach())
        return coarse_loss + fine_loss + 0.1 * commitment


class Model(nn.Module):
    """Hierarchical coarse-to-fine autoregressive financial record forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        code_bits: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.code_bits = code_bits
        self.half_bits = code_bits // 2
        self.tokenizer = HierarchicalTokenizer(enc_in, d_model, code_bits)
        self.coarse_embedding = nn.Linear(self.half_bits, d_model, bias=False)
        self.fine_embedding = nn.Linear(self.half_bits, d_model, bias=False)
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.blocks = nn.ModuleList(
            [CausalBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        vocabulary = 2**self.half_bits
        self.coarse_head = nn.Linear(d_model, vocabulary)
        self.fine_context = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.fine_head = nn.Linear(d_model, vocabulary)
        self.register_buffer("subtoken_codebook", _binary_codebook(self.half_bits))
        self.last_coarse_probabilities: list[torch.Tensor] = []
        self.last_fine_probabilities: list[torch.Tensor] = []

    def _embed_bits(self, bits: torch.Tensor) -> torch.Tensor:
        coarse = self.coarse_embedding(bits[..., : self.half_bits])
        fine = self.fine_embedding(bits[..., self.half_bits :])
        return self.fusion(torch.cat((coarse, fine), -1))

    def _expected_bits(self, probabilities: torch.Tensor) -> torch.Tensor:
        return probabilities @ self.subtoken_codebook.to(probabilities)

    def tokenizer_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Equation (2): coarse, full reconstruction, and BSQ commitment."""
        mean = x.mean(1, keepdim=True)
        scale = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return self.tokenizer.hierarchical_reconstruction_loss((x - mean) / scale)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        mean = x_enc.mean(1, keepdim=True).detach()
        scale = x_enc.std(1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = (x_enc - mean) / scale
        history_bits, _ = self.tokenizer.encode(normalized)
        hidden = self._embed_bits(history_bits)
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            hidden, cache = block.prefill(hidden)
            caches.append(cache)

        predictions: list[torch.Tensor] = []
        coarse_records: list[torch.Tensor] = []
        fine_records: list[torch.Tensor] = []
        for _ in range(self.pred_len):
            history_state = hidden[:, -1:]
            coarse_probability = torch.softmax(self.coarse_head(history_state), -1)
            coarse_bits = self._expected_bits(coarse_probability)
            coarse_query = self.coarse_embedding(coarse_bits)
            fine_state, _ = self.fine_context(coarse_query, hidden, hidden, need_weights=False)
            fine_probability = torch.softmax(self.fine_head(fine_state), -1)
            fine_bits = self._expected_bits(fine_probability)
            bits = torch.cat((coarse_bits, fine_bits), -1)
            predictions.append(self.tokenizer.decoder(bits))
            coarse_records.append(coarse_probability)
            fine_records.append(fine_probability)

            token = self._embed_bits(bits)
            new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for block, cache in zip(self.blocks, caches, strict=True):
                token, new_cache = block.step(token, cache)
                new_caches.append(new_cache)
            caches = new_caches
            hidden = torch.cat((hidden, token), dim=1)

        self.last_coarse_probabilities = coarse_records
        self.last_fine_probabilities = fine_records
        forecast = torch.cat(predictions, dim=1)
        return forecast * scale + mean

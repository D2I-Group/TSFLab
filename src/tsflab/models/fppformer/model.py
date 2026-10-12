"""Local FPPformer implementation (paper Section IV, Figs. 2-5, Table I).

Independent rewrite from the paper; the pinned official repository
(Apache-2.0) was read only to resolve omissions such as the shared key/value
projection, the single-head attention, the norm placement, the decoder query
initialization, and the linear encoder projection that is added to the
decoder output.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.embed import PositionalEmbedding
from tsflab.models._components.revin import RevIN


def diagonal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[length, length]`` mask of the query-key diagonal (Fig. 5)."""
    return torch.eye(length, dtype=torch.bool, device=device)


class SharedKVAttention(nn.Module):
    """Single-head scaled dot-product attention over the second-to-last axis.

    Keys and values come from the same memory through one shared projection.
    With ``mask_diagonal`` each token can only be expressed by the other tokens
    (diagonal-masked self-attention, Section IV-D).
    """

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, mask_diagonal: bool = False) -> torch.Tensor:
        q = self.query(queries)
        kv = self.key_value(memory)
        scores = q @ kv.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if mask_diagonal:
            scores = scores.masked_fill(diagonal_mask(scores.shape[-1], scores.device), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(weights @ kv)


class FeedForward(nn.Module):
    """Position-wise ``Linear(4x) -> GELU -> Linear`` over the flattened patch."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.activation = nn.GELU()
        self.down = nn.Linear(4 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.activation(self.up(x)))


class EncoderStage(nn.Module):
    """Bottom-up encoder stage (Fig. 2, Table I).

    Input ``[B, V, S, P, D]`` (S patches of P elements). Diagonal-masked
    element-wise self-attention inside every patch, reshape to patch tokens of
    width ``P * D``, diagonal-masked patch-wise self-attention, and a feed
    forward layer, each with a residual and post LayerNorm. Returns
    ``[B, V, S, P * D]``.
    """

    def __init__(self, patch_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        width = patch_size * d_model
        self.element_attention = SharedKVAttention(d_model, dropout)
        self.patch_attention = SharedKVAttention(width, dropout)
        self.feed_forward = FeedForward(width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.element_attention(x, x, mask_diagonal=True))
        h = self.norm1(x.flatten(start_dim=-2))
        h = self.norm2(h + self.dropout(self.patch_attention(h, h, mask_diagonal=True)))
        return self.norm3(h + self.dropout(self.feed_forward(h)))


class DecoderStage(nn.Module):
    """Top-down decoder stage (Fig. 2, Section IV-B/C).

    Input queries ``[B, V, S, P, D]`` and the same-resolution encoder map
    ``[B, V, S_enc, P * D]`` (lateral connection). Patch-wise cross-attention
    comes first, then unmasked element-wise self-attention inside each patch,
    then the feed forward layer; residuals with post LayerNorm. Returns
    ``[B, V, S, P, D]``.
    """

    def __init__(self, patch_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        width = patch_size * d_model
        self.cross_attention = SharedKVAttention(width, dropout)
        self.element_attention = SharedKVAttention(d_model, dropout)
        self.feed_forward = FeedForward(width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = x.flatten(start_dim=-2)
        h = self.norm1(h + self.dropout(self.cross_attention(h, memory)))
        elements = h.view(shape)
        attended = self.element_attention(elements, elements).flatten(start_dim=-2)
        h = self.norm2(h + self.dropout(attended))
        h = self.norm3(h + self.dropout(self.feed_forward(h)))
        return h.view(shape)


def merge_patches(x: torch.Tensor, d_model: int) -> torch.Tensor:
    """``[B, V, S, P * D]`` to ``[B, V, S / 2, 2P, D]``: adjacent patches merge (encoder)."""
    batch, variates, patches, width = x.shape
    return x.reshape(batch, variates, patches // 2, 2 * width // d_model, d_model)


def split_patches(x: torch.Tensor) -> torch.Tensor:
    """``[B, V, S, P, D]`` to ``[B, V, 2S, P / 2, D]``: every patch splits in two (decoder)."""
    batch, variates, patches, size, d_model = x.shape
    return x.reshape(batch, variates, 2 * patches, size // 2, d_model)


class Model(nn.Module):
    """FPPformer: bottom-up patch-merging encoder and top-down patch-splitting decoder.

    The output is the RevIN-denormalized sum of a linear projection of the
    final encoder map and the projected decoder output.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        patch_size: int = 6,
        num_stages: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        top_patch = patch_size * 2 ** (num_stages - 1)
        if seq_len % top_patch != 0 or seq_len // top_patch < 2:
            raise ValueError(
                f"seq_len={seq_len} must be a multiple of patch_size * 2**(num_stages - 1)={top_patch} "
                "with at least two patches at the top stage"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.top_patch = top_patch
        # Decoder length padded up to a whole number of top-stage patches.
        self.decoder_len = math.ceil(pred_len / top_patch) * top_patch

        self.revin = RevIN(enc_in, affine=False)
        self.value_embedding = nn.Linear(1, d_model)
        self.position = PositionalEmbedding(d_model, max_len=seq_len + pred_len)
        self.encoders = nn.ModuleList(
            EncoderStage(patch_size * 2**i, d_model, dropout) for i in range(num_stages)
        )
        self.decoders = nn.ModuleList(
            DecoderStage(patch_size * 2 ** (num_stages - 1 - i), d_model, dropout) for i in range(num_stages)
        )
        hidden = max(2 * seq_len, 2 * pred_len)
        self.encoder_point_projection = nn.Linear(d_model, 1)
        self.encoder_time_projection = nn.Sequential(nn.Linear(seq_len, hidden), nn.Linear(hidden, pred_len))
        self.decoder_projection = nn.Linear(d_model, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Channel- and element-independent embedding ``[B, L, V] -> [B, V, L, D]`` (Fig. 4)."""
        values = self.value_embedding(x.unsqueeze(-1))
        positions = self.position.pe[:, : x.shape[1]].unsqueeze(2)
        return (values + positions).transpose(1, 2)

    def encode(self, embedded: torch.Tensor) -> list[torch.Tensor]:
        """Bottom-up stages; returns every stage map ``[B, V, S_i, P_i * D]`` (lateral outputs)."""
        batch, variates, length, _ = embedded.shape
        h = embedded.reshape(batch, variates, length // self.patch_size, self.patch_size, self.d_model)
        maps = []
        for index, stage in enumerate(self.encoders):
            out = stage(h)
            maps.append(out)
            if index < self.num_stages - 1:
                h = merge_patches(out, self.d_model)
        return maps

    def decoder_queries(self, embedded: torch.Tensor) -> torch.Tensor:
        """Future position embeddings, prefixed by the latest embedded inputs when padding is needed."""
        batch, variates = embedded.shape[:2]
        future = self.position.pe[:, self.seq_len : self.seq_len + self.pred_len]
        queries = future.unsqueeze(1).expand(batch, variates, self.pred_len, self.d_model)
        padding = self.decoder_len - self.pred_len
        if padding > 0:
            queries = torch.cat([embedded[:, :, -padding:], queries], dim=2)
        return queries.reshape(
            batch, variates, self.decoder_len // self.top_patch, self.top_patch, self.d_model
        )

    def decode(self, queries: torch.Tensor, maps: list[torch.Tensor]) -> torch.Tensor:
        """Top-down stages, each attending to the encoder map of the same patch size."""
        h = queries
        for index, stage in enumerate(self.decoders):
            h = stage(h, maps[-1 - index])
            if index < self.num_stages - 1:
                h = split_patches(h)
        batch, variates = h.shape[:2]
        return h.reshape(batch, variates, self.decoder_len, self.d_model)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(
                f"expected input [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        x = self.revin(x_enc, "norm")
        embedded = self.embed(x)
        maps = self.encode(embedded)
        batch, variates = embedded.shape[:2]
        encoded = maps[-1].reshape(batch, variates, self.seq_len, self.d_model)
        linear = self.encoder_time_projection(self.encoder_point_projection(encoded).squeeze(-1))
        decoded = self.decoder_projection(self.decode(self.decoder_queries(embedded), maps)).squeeze(-1)
        forecast = linear + decoded[:, :, -self.pred_len :]
        return self.revin(forecast.transpose(1, 2), "denorm")

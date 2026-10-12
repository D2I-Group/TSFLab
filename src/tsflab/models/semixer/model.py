"""Clean-room SEMixer: Random Attention Mechanism and progressive multiscale mixing.

Paper: "SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and
Long-term Time Series Forecasting" (arXiv:2602.16220, WWW 2026).

Every scale patchifies the normalized history with its own patch length
``patch_len * scale`` and stride ``stride * scale`` (Sec. 3.1, Eqs. 1-3). Each
patch token is linearly embedded and combined with a learnable per-scale
position table (Eq. 4). The Random Attention Mechanism (RAM, Sec. 3.2)
replaces a learned attention matrix with a matrix sampled once per forward
call from ``Bernoulli(1 - connection_probability)``: at training time it
randomly zeroes patch-to-patch links before an all-ones aggregation, and at
evaluation time it uses the closed-form dropout-ensemble approximation
``(1 - connection_probability) * ones`` (Eqs. 6-7) instead of averaging many
sampled masks explicitly. The Multiscale Progressive Mixing Chain (MPMC,
Sec. 3.3, Algorithm 1) mixes the finest scale on its own, then repeatedly
concatenates the previous scale's mixed tokens with the next scale's raw
tokens and mixes the pair, keeping only the suffix belonging to the new
scale. All per-scale outputs are finally concatenated along the patch axis,
reduced to a fixed token budget, and flattened into a linear forecast head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN


def _parse_scale_factors(raw: str) -> tuple[int, ...]:
    parts = [chunk for chunk in raw.lower().split("x") if chunk]
    if not parts:
        raise ValueError("scale_factors must contain at least one scale")
    try:
        scales = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"scale_factors must be integers separated by 'x': {raw!r}") from error
    if scales[0] != 1:
        raise ValueError("the finest scale factor must be 1")
    if list(scales) != sorted(scales) or len(set(scales)) != len(scales):
        raise ValueError("scale_factors must be strictly increasing")
    return scales


class InterPatchMixing(nn.Module):
    """Two-layer MLP mixing across the patch axis (paper Sec. 3.3)."""

    def __init__(self, patch_num: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(patch_num, patch_num)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(patch_num, patch_num)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.dropout1(self.act(self.fc1(values)))
        return self.dropout2(self.fc2(values))


class IntraPatchMixing(nn.Module):
    """Two-layer MLP mixing across the embedding axis (paper Sec. 3.3)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.dropout1(self.act(self.fc1(values)))
        return self.dropout2(self.fc2(values))


class TemporalMixingBlock(nn.Module):
    """RAM-gated inter/intra patch mixing (paper Sec. 3.2-3.3, Eqs. 5-7)."""

    def __init__(self, patch_num: int, d_model: int, connection_probability: float, mixing_dropout: float) -> None:
        super().__init__()
        self.patch_num = patch_num
        self.connection_probability = connection_probability
        self.inter_patch_mixing = InterPatchMixing(patch_num, mixing_dropout)
        self.intra_patch_mixing = IntraPatchMixing(d_model, mixing_dropout)
        self.last_adjacency: torch.Tensor | None = None

    def _random_attention(self, values: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(self.patch_num, self.patch_num, device=values.device, dtype=values.dtype)
        if self.training:
            keep = torch.rand(self.patch_num, self.patch_num, device=values.device) > self.connection_probability
            adjacency = ones * keep.to(values.dtype)
        else:
            adjacency = ones * (1.0 - self.connection_probability)
        self.last_adjacency = adjacency
        return torch.matmul(adjacency.unsqueeze(0), values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [batch*channels, patch_num, d_model]
        residual = values
        mixed = self._random_attention(values).transpose(1, 2)
        inter = self.inter_patch_mixing(mixed).transpose(1, 2)
        stage_one = inter + residual
        stage_two = self.intra_patch_mixing(stage_one) + stage_one + residual
        return stage_two


class ScaleEmbedding(nn.Module):
    """Patchify one scale and embed patches with a learnable position table."""

    def __init__(self, seq_len: int, patch_len: int, stride: int, scale: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.scale = scale
        self.patch_len = patch_len * scale
        self.stride = stride * scale
        self.padding = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = (seq_len - self.patch_len) // self.stride + 1 + 1
        if self.patch_num < 1:
            raise ValueError(f"scale factor {scale} leaves no patches for seq_len={seq_len}")
        self.value_embedding = nn.Linear(self.patch_len, d_model)
        self.position_embedding = positional_encoding("zeros", True, self.patch_num, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [batch, channels, seq_len] -> [batch*channels, patch_num, d_model]
        batch, channels, _ = values.shape
        padded = self.padding(values)
        patches = padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch * channels, self.patch_num, self.patch_len)
        embedded = self.value_embedding(patches) + self.position_embedding
        return self.dropout(embedded)


class Model(nn.Module):
    """Forecast-only SEMixer with RevIN, RAM-gated mixing, and MPMC scale chaining."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        d_model: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        scale_factors: str = "1x2x4x8",
        reduce_dim: int = 64,
        eib_num: int = 1,
        eib_num_1scale: int = 1,
        connection_probability: float = 0.85,
        dropout: float = 0.05,
        mixing_dropout: float = 0.1,
        head_dropout: float = 0.0,
        affine: bool = False,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths, channels, widths, and layer counts must be positive")
        scales = _parse_scale_factors(scale_factors)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.scales = scales

        self.revin = RevIN(enc_in, affine=affine, subtract_last=subtract_last)
        self.embeddings = nn.ModuleList(
            [ScaleEmbedding(seq_len, patch_len, stride, scale, d_model, dropout) for scale in scales]
        )
        patch_nums = [embedding.patch_num for embedding in self.embeddings]

        # Finest scale mixes alone; each coarser scale mixes concatenated with
        # the previous scale's mixed tail (Algorithm 1).
        self.finest_chain = nn.ModuleList(
            [TemporalMixingBlock(patch_nums[0], d_model, connection_probability, mixing_dropout) for _ in range(eib_num_1scale)]
        )
        self.coarser_chains = nn.ModuleList()
        running_patch_num = patch_nums[0]
        for patch_num in patch_nums[1:]:
            paired_num = running_patch_num + patch_num
            self.coarser_chains.append(
                nn.ModuleList(
                    [TemporalMixingBlock(paired_num, d_model, connection_probability, mixing_dropout) for _ in range(eib_num)]
                )
            )
            running_patch_num = patch_num

        self.reduce = nn.Linear(sum(patch_nums), reduce_dim)
        self.head = FlattenForecastHead(False, enc_in, d_model * reduce_dim, pred_len, head_dropout)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.channels})")
        batch = x_enc.shape[0]
        normalized = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, seq_len]

        scale_tokens = [embedding(normalized) for embedding in self.embeddings]

        mixed = scale_tokens[0]
        for block in self.finest_chain:
            mixed = block(mixed)
        finished = [mixed]
        for chain, raw_tokens in zip(self.coarser_chains, scale_tokens[1:]):
            paired = torch.cat([mixed, raw_tokens], dim=1)
            for block in chain:
                paired = block(paired)
            mixed = paired[:, -raw_tokens.shape[1]:, :]
            finished.append(mixed)

        # Exposed for structural tests: one patch count per scale, after MPMC
        # keeps only the newest scale's suffix out of each pairwise mix.
        self.last_scale_patch_nums = [tokens.shape[1] for tokens in finished]

        concatenated = torch.cat(finished, dim=1)  # [B*C, sum(patch_num), d_model]
        reduced = self.reduce(concatenated.transpose(1, 2)).transpose(1, 2)
        reduced = reduced.reshape(batch, self.channels, reduced.shape[-2], reduced.shape[-1])

        forecast = self.head(reduced).transpose(1, 2)
        return self.revin(forecast, "denorm")

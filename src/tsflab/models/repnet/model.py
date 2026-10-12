"""REP-Net: Representation, Memory and Projection modules for forecasting.

Independent implementation of Leppich et al., "Decomposing the Time Series
Forecasting Pipeline: A Modular Approach for Time Series Representation,
Information Extraction, and Projection" (arXiv 2507.05891), checked against
RobertLeppich/REP-Net@df3b2983 (no license file; nothing copied).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import adapt_tslib_marks, tslib_time_feature_dimension
from tsflab.models._components.revin import RevIN

REPRESENTATIONS = (
    "single_linear",
    "linear",
    "glu",
    "simple_linear",
    "double_linear",
    "double_linear_glu",
    "cnn_1",
    "cnn_2",
    "cnn_3",
)
TIME_EMBEDDINGS = ("posEmb", "timeF", "tempEmb")


def patch_count(seq_len: int, cover: int, stride: int) -> int:
    """Number of sub-sequences of length ``cover`` taken every ``stride`` steps."""
    return (seq_len - cover) // stride + 1


# Linear-family value encoders by kind; the CNN family runs ``ValueEncoder._cnn``.
_LINEAR_FORWARD = {
    "single_linear": lambda m, x: m.proj(x),
    "linear": lambda m, x: m.outer(F.gelu(m.inner(x))),
    "glu": lambda m, x: F.glu(m.outer(F.gelu(m.inner(x))), dim=-1),
    "simple_linear": lambda m, x: x @ m.matrix,
    "double_linear": lambda m, x: m.outer(x @ m.matrix),
    "double_linear_glu": lambda m, x: F.glu(m.outer(x @ m.matrix), dim=-1),
}


class ValueEncoder(nn.Module):
    """Embed one sub-sequence ``[B, P, F, n]`` to ``[B, P, F, e]`` (Sec. 2.2, Appendix C).

    Linear family (E1/E2) acts on the last axis with weights shared over
    variables; the CNN family (E3) uses per-variable grouped ``k = 3`` convolutions
    over the ``n`` samples and a linear read-out that collapses them.
    """

    def __init__(self, kind: str, length: int, encoding: int, num_vars: int) -> None:
        super().__init__()
        if kind not in REPRESENTATIONS:
            raise ValueError(f"unknown representation {kind!r}")
        self.kind = kind
        self.encoding = encoding
        self.num_vars = num_vars
        if kind == "single_linear":
            self.proj = nn.Linear(length, encoding)
        elif kind in ("linear", "glu"):
            self.inner = nn.Linear(length, 2 * encoding, bias=False)
            self.outer = nn.Linear(2 * encoding, 2 * encoding if kind == "glu" else encoding)
        elif kind in ("simple_linear", "double_linear", "double_linear_glu"):
            self._build_matrix(kind, length, encoding)
        else:
            self._build_cnn(kind, length, encoding, num_vars)

    def _build_matrix(self, kind: str, length: int, encoding: int) -> None:
        self.matrix = nn.Parameter(torch.randn(length, encoding))
        if kind == "double_linear":
            self.outer = nn.Linear(encoding, encoding)
        elif kind == "double_linear_glu":
            self.outer = nn.Linear(encoding, 2 * encoding)

    def _build_cnn(self, kind: str, length: int, encoding: int, num_vars: int) -> None:
        width = encoding * num_vars
        self.conv1 = nn.Conv1d(num_vars, width, 3, padding=1, groups=num_vars)
        if kind != "cnn_1":
            self.conv2 = nn.Conv1d(width, width, 3, padding=1, groups=num_vars)
        self.readout = nn.Linear(length, 1)

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        # Zero padding then a stride-1 max over three samples keeps the length.
        return F.max_pool1d(F.pad(x, (1, 1)), kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = _LINEAR_FORWARD.get(self.kind)
        return self._cnn(x) if linear is None else linear(self, x)

    def _cnn(self, x: torch.Tensor) -> torch.Tensor:
        kind = self.kind
        batch, patches, variables, length = x.shape
        h = self.conv1(x.reshape(batch * patches, variables, length))
        if kind == "cnn_2":
            h = self.conv2(F.gelu(h))
            h = F.gelu(h)
        elif kind == "cnn_3":
            h = self._pool(self.conv2(self._pool(h)))
        h = self.readout(h).squeeze(-1)  # [B * P, F * e], variable-major groups
        return h.reshape(batch, patches, variables, self.encoding)


class TimeEncoding(nn.Module):
    """Time embedding of the history (Sec. 2.2): sum of the selected encodings, standardized.

    ``posEmb`` is the fixed table ``0.5 + 0.5 sin/cos``; ``timeF`` a bias-free
    linear map of continuous calendar features, min-max scaled; ``tempEmb`` learned
    month/day/weekday/hour(/quarter-hour) tables, min-max scaled. Mark-based
    scalings and the final standardization are taken per sample.
    """

    def __init__(self, kinds: Sequence[str], size: int, seq_len: int, freq: str) -> None:
        super().__init__()
        self.kinds = tuple(kinds)
        self.freq = freq
        if "posEmb" in self.kinds:
            if size % 2:
                raise ValueError("posEmb needs an even time_embedding_size")
            position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, size, 2, dtype=torch.float32) * (-math.log(10000.0) / size))
            table = torch.zeros(seq_len, size)
            table[:, 0::2] = 0.5 + 0.5 * torch.sin(position * div)
            table[:, 1::2] = 0.5 + 0.5 * torch.cos(position * div)
            self.register_buffer("position_table", table, persistent=False)
        if "timeF" in self.kinds:
            self.time_feature = nn.Linear(tslib_time_feature_dimension(freq), size, bias=False)
        if "tempEmb" in self.kinds:
            self.month = nn.Embedding(13, size)
            self.day = nn.Embedding(32, size)
            self.weekday = nn.Embedding(7, size)
            self.hour = nn.Embedding(24, size)
            self.minute = nn.Embedding(4, size) if freq.lower() == "t" else None

    @staticmethod
    def _min_max(x: torch.Tensor) -> torch.Tensor:
        low = x.amin(dim=(1, 2), keepdim=True)
        high = x.amax(dim=(1, 2), keepdim=True)
        return (x - low) / (high - low).clamp_min(1e-12)

    @staticmethod
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-5)

    def calendar(self, marks: torch.Tensor) -> torch.Tensor:
        """Learned calendar embedding from raw ``[year, month, day, weekday, hour, minute]`` marks."""
        raw = marks.shape[-1] == 6
        idx = adapt_tslib_marks(marks, embed_type="fixed", freq=self.freq).long()
        out = self.month(idx[..., 0]) + self.day(idx[..., 1]) + self.weekday(idx[..., 2]) + self.hour(idx[..., 3])
        if self.minute is not None:
            out = out + self.minute(idx[..., 4] // 15 if raw else idx[..., 4])
        return self._min_max(out)

    def _mark_parts(self, marks: torch.Tensor) -> list[torch.Tensor]:
        parts = []
        if "timeF" in self.kinds:
            features = adapt_tslib_marks(marks, embed_type="timeF", freq=self.freq)
            parts.append(self._min_max(self.time_feature(features.float())))
        if "tempEmb" in self.kinds:
            parts.append(self.calendar(marks))
        return parts

    def forward(self, marks: torch.Tensor | None, batch: int) -> torch.Tensor:
        parts = []
        if "posEmb" in self.kinds:
            parts.append(self.position_table.expand(batch, -1, -1))
        if ("timeF" in self.kinds or "tempEmb" in self.kinds) and marks is None:
            raise ValueError("REPNet timeF/tempEmb time embeddings require x_mark_enc")
        parts += self._mark_parts(marks)
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return self._standardize(total)


class Representation(nn.Module):
    """K patch extractors with value (and optional time) encoders, concatenated over patches."""

    def __init__(
        self,
        seq_len: int,
        num_vars: int,
        extractors: Sequence[Sequence[int]],
        encoding: int,
        kind: str,
        time_kinds: Sequence[str],
        time_size: int,
        freq: str,
    ) -> None:
        super().__init__()
        self.extractors = [tuple(int(v) for v in spec) for spec in extractors]
        self.patch_counts = self._patch_counts(seq_len)
        self.time = TimeEncoding(time_kinds, time_size, seq_len, freq) if time_kinds else None
        value_size = encoding // 2 if self.time is not None else encoding
        self.value_encoders = nn.ModuleList(
            ValueEncoder(kind, math.ceil(cover / dilation), value_size, num_vars)
            for cover, dilation, _ in self.extractors
        )
        self.time_encoders = self._time_encoders(encoding, time_size)

    def _patch_counts(self, seq_len: int) -> list[int]:
        counts = [patch_count(seq_len, cover, stride) for cover, _, stride in self.extractors]
        if min(counts) < 1:
            raise ValueError("every patch extractor cover size must be <= seq_len")
        return counts

    def _time_encoders(self, encoding: int, time_size: int) -> nn.ModuleList:
        return nn.ModuleList(
            nn.Linear(cover * time_size, encoding // 2) for cover, _, _ in self.extractors
        ) if self.time is not None else nn.ModuleList()

    @staticmethod
    def extract(x: torch.Tensor, cover: int, dilation: int, stride: int) -> torch.Tensor:
        """``[B, T, F] -> [B, P, F, ceil(cover / dilation)]``: windows every ``stride`` steps, then every ``dilation``-th sample."""
        return x.unfold(1, cover, stride)[..., ::dilation]

    def forward(self, x: torch.Tensor, marks: torch.Tensor | None) -> torch.Tensor:
        time = self.time(marks, x.shape[0]) if self.time is not None else None
        parts = []
        for k, (cover, dilation, stride) in enumerate(self.extractors):
            value = self.value_encoders[k](self.extract(x, cover, dilation, stride))
            if time is not None:
                # Time-informed patch: the time sub-sequence (no dilation) is embedded once
                # per patch and shared by every variable.
                window = time.unfold(1, cover, stride).flatten(-2)
                stamp = self.time_encoders[k](window).unsqueeze(2).expand(-1, -1, value.shape[2], -1)
                value = torch.cat([value, stamp], dim=-1)
            parts.append(value)
        return torch.cat(parts, dim=1)  # [B, S, F, e]


class PatchAttention(nn.Module):
    """Multi-head softmax attention across the ``S`` patches of each variable."""

    def __init__(self, encoding: int, heads: int, dropout: float) -> None:
        super().__init__()
        if encoding % heads:
            raise ValueError("encoding_size must be divisible by n_heads")
        self.heads = heads
        self.query = nn.Linear(encoding, encoding)
        self.key = nn.Linear(encoding, encoding)
        self.value = nn.Linear(encoding, encoding)
        self.out = nn.Linear(encoding, encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, variables, encoding = x.shape
        tokens = x.transpose(1, 2)  # [B, F, S, e]

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, variables, patches, self.heads, -1).transpose(2, 3)

        q, k, v = split(self.query(tokens)), split(self.key(tokens)), split(self.value(tokens))
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        mixed = (self.dropout(weights) @ v).transpose(2, 3).reshape(batch, variables, patches, encoding)
        return self.out(mixed).transpose(1, 2)


class MemoryBlock(nn.Module):
    """One memory module (Sec. 2.3): positional mixing, optional attention, GLU and feature mixing.

    Each sub-block is pre-LayerNorm with a residual connection.
    """

    def __init__(
        self,
        num_patches: int,
        encoding: int,
        num_vars: int,
        heads: int,
        dropout: float,
        attention: bool,
        glu: bool,
        joint_features: bool,
    ) -> None:
        super().__init__()
        self.pos_norm = nn.LayerNorm(num_patches)
        self.positional = nn.Linear(num_patches, num_patches)
        self.attn_norm = nn.LayerNorm(encoding) if attention else None
        self.attention = PatchAttention(encoding, heads, dropout) if attention else None
        self.glu_norm = nn.LayerNorm(encoding)
        self.glu = nn.Linear(encoding, 2 * encoding) if glu else None
        self.joint_features = joint_features
        width = encoding * num_vars if joint_features else encoding
        self.feature = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def mix_positions(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm and Linear over the patch axis of ``[B, S, F, e]``, ReLU, dropout."""
        h = x.transpose(1, -1)  # [B, e, F, S]
        h = self.positional(self.pos_norm(h))
        return self.dropout(F.relu(h)).transpose(1, -1)

    def mix_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.glu_norm(x)
        if self.glu is not None:
            h = self.dropout(F.glu(self.glu(h), dim=-1))
        if self.joint_features:
            h = self.feature(h.flatten(-2)).reshape(x.shape)
        else:
            h = self.feature(h)
        return self.dropout(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix_positions(x)
        if self.attention is not None:
            x = x + self.dropout(F.gelu(self.attention(self.attn_norm(x))))
        return x + self.mix_features(x)


class Projection(nn.Module):
    """Per-extractor optional LSTM over patches and bias-free linear head; heads are summed (Sec. 2.4)."""

    def __init__(self, patch_counts: Sequence[int], encoding: int, pred_len: int, lstm_layers: int) -> None:
        super().__init__()
        self.patch_counts = list(patch_counts)
        self.lstms = nn.ModuleList(
            nn.LSTM(encoding, encoding, num_layers=lstm_layers, batch_first=True) for _ in self.patch_counts
        ) if lstm_layers > 0 else nn.ModuleList()
        self.heads = nn.ModuleList(nn.Linear(encoding * n, pred_len, bias=False) for n in self.patch_counts)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch, _, variables, encoding = memory.shape
        total = 0
        for k, part in enumerate(torch.split(memory, self.patch_counts, dim=1)):
            seq = part.transpose(1, 2).reshape(batch * variables, part.shape[1], encoding)
            if len(self.lstms):
                seq, _ = self.lstms[k](seq)
            total = total + self.heads[k](seq.reshape(batch, variables, -1))
        return total.transpose(1, 2)  # [B, H, F]


class Model(nn.Module):
    """REP-Net forecaster with the four-input TSFLab interface."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_extractors: Sequence[Sequence[int]] = ((3, 1, 1), (10, 2, 4), (15, 3, 5)),
        encoding_size: int = 16,
        representation: str = "linear",
        time_embedding: Sequence[str] = ("posEmb",),
        time_embedding_size: int = 16,
        freq: str = "h",
        memory_layers: int = 3,
        attention: bool = False,
        n_heads: int = 16,
        glu: bool = True,
        joint_feature_mixing: bool = False,
        lstm_layers: int = 0,
        dropout: float = 0.3,
        revin: bool = True,
    ) -> None:
        super().__init__()
        unknown = set(time_embedding) - set(TIME_EMBEDDINGS)
        if unknown:
            raise ValueError(f"unknown time embeddings: {sorted(unknown)}")
        if time_embedding and encoding_size % 2:
            raise ValueError("encoding_size must be even when a time embedding is used")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = RevIN(enc_in, affine=True, subtract_last=False, enabled=revin)
        self.representation = Representation(
            seq_len, enc_in, patch_extractors, encoding_size, representation,
            tuple(time_embedding), time_embedding_size, freq,
        )
        num_patches = sum(self.representation.patch_counts)
        self.memory = nn.ModuleList(
            MemoryBlock(num_patches, encoding_size, enc_in, n_heads, dropout, attention, glu, joint_feature_mixing)
            for _ in range(memory_layers)
        )
        self.projection = Projection(self.representation.patch_counts, encoding_size, pred_len, lstm_layers)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.revin(x_enc, "norm")
        h = self.representation(x, x_mark_enc)
        for block in self.memory:
            h = block(h)
        return self.revin(self.projection(h), "denorm")

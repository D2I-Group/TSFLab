"""Local GCformer: global convolution branch + local PatchTST branch + cross-attention decoder.

Paper map (Zhao et al., 2023, arXiv 2306.08325): instance normalization (RevIN,
Eqs. 13-15); the global branch is a global convolution ``y = u * k`` computed by FFT
(Eqs. 1-2) whose kernel is assembled from multi-scale sub-kernels with decaying
weights (Gconv_msk, Eq. 3), followed by a linear map to the horizon (Eq. 8); the
local branch is a channel-independent PatchTST over the most recent ``context_len``
steps (Eq. 9); the decoder maps both forecasts to hidden tokens (Eqs. 10-11) and
fuses them with cross-attention (Eq. 12) over channel and token views, plus
learnable shortcuts of the two branch forecasts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import (
    AttentionLayer,
    FullAttention,
    ProbAttention,
)
from tsflab.models._inputs import check_window


def multiscale_kernel_count(seq_len: int, kernel_dim: int) -> int:
    """Sub-kernel count ``1 + ceil(log2(L / kernel_dim))`` (at least one)."""
    return max(1, 1 + math.ceil(math.log2(seq_len / kernel_dim)))


class MultiScaleGlobalConv(nn.Module):
    """Gconv_msk (Eq. 3): bidirectional global convolution with a multi-scale kernel.

    Each sub-kernel has ``kernel_dim`` free taps per (head, variate); sub-kernel ``i``
    is linearly upsampled by ``2 ** max(0, i - 1)`` and weighted by
    ``decay ** (n - i - 1)`` so far lags get geometrically smaller weights; the
    concatenation is normalised by its initial L2 norm, cut to the sequence length,
    and split into a causal half and an anti-causal (flipped) half. The FFT
    convolution (Eq. 2) is followed by a ``D`` skip term, GELU, and a linear map from
    ``heads * variates`` back to ``variates``. Input and output are ``[B, L, H]``.
    """

    def __init__(self, width: int, seq_len: int, heads: int, kernel_dim: int = 32, decay: float = 2.0) -> None:
        super().__init__()
        self.width, self.heads, self.seq_len = width, heads, seq_len
        self.kernel_dim, self.decay = kernel_dim, float(decay)
        self.num_scales = multiscale_kernel_count(seq_len, kernel_dim)
        # Two kernel sets: causal (first ``heads``) and anti-causal (last ``heads``).
        self.sub_kernels = nn.ParameterList(
            nn.Parameter(torch.randn(2 * heads, width, kernel_dim)) for _ in range(self.num_scales)
        )
        self.skip = nn.Parameter(torch.randn(heads, width))
        self.output_linear = nn.Linear(heads * width, width)
        with torch.no_grad():
            self.register_buffer("kernel_norm", self.assembled_kernel().norm(dim=-1, keepdim=True))

    def assembled_kernel(self) -> torch.Tensor:
        """Concatenated, decay-weighted multi-scale kernel ``[2 * heads, width, K]``."""
        pieces = []
        for i, sub_kernel in enumerate(self.sub_kernels):
            upsampled = F.interpolate(sub_kernel, scale_factor=2 ** max(0, i - 1), mode="linear")
            pieces.append(upsampled * self.decay ** (self.num_scales - i - 1))
        return torch.cat(pieces, dim=-1)

    def kernel(self, length: int) -> torch.Tensor:
        """Normalised two-sided kernel of length ``2 * length``: ``[heads, width, 2L]``."""
        k = self.assembled_kernel()
        k = k[..., :length] if k.shape[-1] >= length else F.pad(k, (0, length - k.shape[-1]))
        k = k / self.kernel_norm
        causal, anti_causal = k[: self.heads], k[self.heads :]
        return F.pad(causal, (0, length)) + F.pad(anti_causal.flip(-1), (length, 0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        u = values.transpose(1, 2)  # [B, H, L]
        # Eq. (2): u * k = F^-1(F(u) F(k)), zero-padded to 2L (no circular wrap).
        spectrum = torch.einsum(
            "bhf,chf->bchf", torch.fft.rfft(u, n=2 * length), torch.fft.rfft(self.kernel(length), n=2 * length)
        )
        y = torch.fft.irfft(spectrum, n=2 * length)[..., :length]
        y = y + torch.einsum("bhl,ch->bchl", u, self.skip)
        y = F.gelu(y.reshape(batch, self.heads * self.width, length))
        return self.output_linear(y.transpose(1, 2))


class ResidualScoreAttention(nn.Module):
    """PatchTST multi-head self-attention that adds the previous layer's pre-softmax scores."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, proj_dropout: float) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        scores = q @ k.transpose(-2, -1) * self.scale
        if prev is not None:
            scores = scores + prev
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.out(out), scores


class PatchEncoderLayer(nn.Module):
    """Post-norm PatchTST encoder layer with BatchNorm and a GELU feed-forward network."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = ResidualScoreAttention(d_model, n_heads, 0.0, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.BatchNorm1d(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.BatchNorm1d(d_model)

    @staticmethod
    def _norm(norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(x, prev)
        x = self._norm(self.attn_norm, x + self.attn_dropout(attended))
        x = self._norm(self.ff_norm, x + self.ff_dropout(self.ff(x)))
        return x, scores


class LocalPatchBranch(nn.Module):
    """Eq. (9): channel-independent PatchTST on the tail ``[B, N', C]`` -> ``[B, T, C]``."""

    def __init__(
        self,
        enc_in: int,
        context_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        head_dropout: float,
        individual: bool,
    ) -> None:
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        # End padding by repeating the last value ``stride`` times adds one patch.
        self.patch_num = (context_len - patch_len) // stride + 2
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position = positional_encoding("zeros", True, self.patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(PatchEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers))
        self.head = FlattenForecastHead(individual, enc_in, d_model * self.patch_num, pred_len, head_dropout)

    def patchify(self, series: torch.Tensor) -> torch.Tensor:
        """``[B, C, N']`` -> ``[B, C, P, patch_len]`` with replicate end padding."""
        padded = F.pad(series, (0, self.stride), mode="replicate")
        return padded.unfold(-1, self.patch_len, self.stride)

    def forward(self, tail: torch.Tensor) -> torch.Tensor:
        batch, _, channels = tail.shape
        patches = self.patchify(tail.transpose(1, 2))
        tokens = self.patch_embedding(patches).reshape(batch * channels, self.patch_num, -1)
        tokens = self.dropout(tokens + self.position)
        scores = None
        for layer in self.layers:
            tokens, scores = layer(tokens, scores)
        features = tokens.reshape(batch, channels, self.patch_num, -1).transpose(2, 3)
        return self.head(features).transpose(1, 2)


class CrossFusion(nn.Module):
    """Eqs. (10)-(12) in one view: both query directions, shared BatchNorm, weighted sum."""

    def __init__(
        self, in_dim: int, hidden: int, n_heads: int, attention: str, fc_dropout: float, atten_bias: float
    ) -> None:
        super().__init__()
        self.project_in = nn.Linear(in_dim, hidden)
        if attention == "full":
            core = FullAttention(False, attention_dropout=0.0, output_attention=False)
        else:
            core = ProbAttention(True, factor=5, attention_dropout=0.0, output_attention=False)
        self.attention = AttentionLayer(core, hidden, n_heads)
        self.norm = nn.BatchNorm1d(hidden)
        self.project_out = nn.Linear(hidden, in_dim)
        self.ff = nn.Sequential(nn.GELU(), nn.Dropout(fc_dropout))
        self.atten_bias = float(atten_bias)

    def attend(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """``ff(Atten(q, k=v=memory)) + memory``, then BatchNorm over the hidden axis."""
        attended = self.attention(query, memory, memory, attn_mask=None)[0]
        fused = self.ff(attended) + memory
        return self.norm(fused.transpose(1, 2)).transpose(1, 2)

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        g, loc = self.project_in(global_view), self.project_in(local_view)
        # Global queries local (paper Eq. 12) and, symmetrically, local queries global.
        mixed = self.atten_bias * self.attend(g, loc) + (1 - self.atten_bias) * self.attend(loc, g)
        return self.project_out(mixed)


class Model(nn.Module):
    """GCformer with the Gconv_msk global branch and the PatchTST local branch."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        context_len: int | None = None,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        e_layers: int = 3,
        dropout: float = 0.05,
        fc_dropout: float = 0.05,
        head_dropout: float = 0.0,
        individual: bool = True,
        kernel_dim: int = 32,
        kernel_decay: float = 2.0,
        h_channel: int = 32,
        h_token: int = 512,
        decoder_attention: str = "full",
        atten_bias: float = 0.5,
        tc_bias: float = 1.0,
        global_bias: float = 0.5,
        local_bias: float = 0.5,
        norm_type: str = "revin",
    ) -> None:
        super().__init__()
        context_len = seq_len if context_len is None else context_len
        if context_len > seq_len or patch_len > context_len:
            raise ValueError("require patch_len <= context_len <= seq_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.context_len, self.norm_type, self.tc_bias = context_len, norm_type, float(tc_bias)
        self.revin = RevIN(enc_in, affine=True)
        self.global_conv = MultiScaleGlobalConv(enc_in, seq_len, n_heads, kernel_dim, kernel_decay)
        self.global_head = nn.Linear(seq_len, pred_len)
        self.local_branch = LocalPatchBranch(
            enc_in,
            context_len,
            pred_len,
            patch_len,
            stride,
            d_model,
            n_heads,
            d_ff,
            e_layers,
            dropout,
            head_dropout,
            individual,
        )
        self.channel_fusion = CrossFusion(enc_in, h_channel, n_heads, decoder_attention, fc_dropout, atten_bias)
        self.token_fusion = CrossFusion(pred_len, h_token, n_heads, decoder_attention, fc_dropout, atten_bias)
        self.ff = nn.Sequential(nn.GELU(), nn.Dropout(fc_dropout))
        # Learnable shortcut weights, initialised to the configured value plus U(0, 0.1).
        self.global_bias = nn.Parameter(torch.rand(1) * 0.1 + global_bias)
        self.local_bias = nn.Parameter(torch.rand(1) * 0.1 + local_bias)

    def branches(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (8)-(9): ``z_global`` from the whole window, ``z_local`` from its tail."""
        z_global = self.global_head(self.global_conv(normalized).transpose(1, 2)).transpose(1, 2)
        z_local = self.local_branch(normalized[:, -self.context_len :])
        return z_global, z_local

    def decode(self, z_global: torch.Tensor, z_local: torch.Tensor) -> torch.Tensor:
        """Channel-view and token-view cross fusion plus the branch shortcuts."""
        channel = self.ff(self.channel_fusion(z_global, z_local))
        token = self.ff(self.token_fusion(z_global.transpose(1, 2), z_local.transpose(1, 2)).transpose(1, 2))
        return (
            self.tc_bias * channel
            + (1 - self.tc_bias) * token
            + self.global_bias * z_global
            + self.local_bias * z_local
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        last = x_enc[:, -1:].detach()
        normalized = self.revin(x_enc, "norm") if self.norm_type == "revin" else x_enc - last
        output = self.decode(*self.branches(normalized))
        return self.revin(output, "denorm") if self.norm_type == "revin" else output + last

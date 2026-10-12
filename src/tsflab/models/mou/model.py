"""MoU: Mixture of Feature Extractors (MoF) + Mixture of Architectures (MoA).

Channel-independent patch forecaster. Per series (after RevIN, end-padding and
unfolding into ``N`` patches of ``patch_len``):

    X_rep = MoF(X_p)  = sum_i R_i(X_p) F_i(X_p)           # paper Eq. 5-7
    R     = top-k renormalized softmax(X_p W_g + SN * softplus(X_p W_noise))
    X'    = MoA(X_rep + position)                         # Mamba -> FFN -> Conv -> Attention

Each ``F_i`` is a linear map ``patch_len -> d_model``. The MoA block applies, in
order: a Mamba mixer with residual and BatchNorm, a GELU feed-forward residual, a
convolution residual (kernel 3), and one post-norm Transformer encoder layer, so
the receptive field widens from selected SSM dependencies to full attention.
A flatten linear head maps the ``d_model * N`` features to the horizon.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class TokenBatchNorm(nn.Module):
    """BatchNorm over the feature axis of ``[batch, tokens, features]``."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MixtureOfFeatureExtractors(nn.Module):
    """Sparsely gated mixture of linear patch extractors (noisy top-k gate)."""

    def __init__(self, patch_len: int, d_model: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.extractors = nn.ModuleList(nn.Linear(patch_len, d_model) for _ in range(experts))
        self.w_gate = nn.Parameter(torch.zeros(patch_len, experts))
        self.w_noise = nn.Parameter(torch.zeros(patch_len, experts))

    def gates(self, patches: torch.Tensor) -> torch.Tensor:
        """Sparse weights ``[..., experts]``: top-k probabilities renormalized to sum to one."""
        logits = patches @ self.w_gate
        if self.training:
            std = F.softplus(patches @ self.w_noise) + 1e-2
            logits = logits + torch.randn_like(logits) * std
        probs = torch.softmax(logits, dim=-1)
        values, indices = probs.topk(self.top_k, dim=-1)
        values = values / (values.sum(-1, keepdim=True) + 1e-6)
        return torch.zeros_like(probs).scatter(-1, indices, values)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches``: ``[batch, tokens, patch_len]`` to ``[batch, tokens, d_model]``."""
        gates = self.gates(patches)
        outputs = torch.stack([extractor(patches) for extractor in self.extractors], dim=-2)
        return (gates.unsqueeze(-1) * outputs).sum(-2)


class PostNormEncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer with BatchNorm (PatchTST style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, attn_dropout: float, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm = TokenBatchNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.ff_drop = nn.Dropout(dropout)
        self.ff_norm = TokenBatchNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x + self.attn_drop(self.attention(x, x, x, need_weights=False)[0]))
        return self.ff_norm(x + self.ff_drop(self.ff(x)))


class MixtureOfArchitecturesBlock(nn.Module):
    """Mamba, feed-forward, convolution and self-attention layers in sequence."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, d_state: int, ffn_expand: int,
        mamba_dropout: float, ffn_dropout: float, block_dropout: float, conv_dropout: float,
        attn_dropout: float, tf_dropout: float,
    ) -> None:
        super().__init__()
        self.mamba = MambaBlock(d_model, 2 * d_model, math.ceil(d_model / 16), 4, d_state)
        self.mamba_drop = nn.Dropout(mamba_dropout)
        self.mamba_norm = TokenBatchNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand), nn.GELU(), nn.Dropout(ffn_dropout),
            nn.Linear(d_model * ffn_expand, d_model),
        )
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv_drop = nn.Dropout(conv_dropout)
        self.conv_norm = TokenBatchNorm(d_model)
        self.block_drop = nn.Dropout(block_dropout)
        self.shared_norm = TokenBatchNorm(d_model)  # reused after the FFN and the conv layer
        self.attention = PostNormEncoderLayer(d_model, n_heads, d_ff, attn_dropout, tf_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mamba_norm(x + self.mamba_drop(self.mamba(x)))
        x = self.shared_norm(x + self.block_drop(self.ffn(x)))
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        conv = self.conv_norm(x + self.conv_drop(conv))
        x = self.shared_norm(x + self.block_drop(conv))
        return self.attention(x)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        e_layers: int = 1,
        num_experts: int = 4,
        top_k: int = 2,
        d_state: int = 21,
        ffn_expand: int = 2,
        dropout: float = 0.1,
        mamba_dropout: float = 0.1,
        ffn_dropout: float = 0.2,
        block_dropout: float = 0.2,
        conv_dropout: float = 0.3,
        attn_dropout: float = 0.0,
        tf_dropout: float = 0.2,
        head_dropout: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        if seq_len < patch_len:
            raise ValueError("seq_len must be at least patch_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        patch_num = (seq_len - patch_len) // stride + 2  # one extra patch from end padding
        self.revin = RevIN(enc_in, affine=False, enabled=use_revin)
        self.pad = nn.ReplicationPad1d((0, stride))
        self.extractors = MixtureOfFeatureExtractors(patch_len, d_model, num_experts, top_k)
        self.position = positional_encoding("zeros", True, patch_num, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            MixtureOfArchitecturesBlock(
                d_model, n_heads, d_ff, d_state, ffn_expand, mamba_dropout, ffn_dropout,
                block_dropout, conv_dropout, attn_dropout, tf_dropout,
            )
            for _ in range(e_layers)
        )
        self.head = FlattenForecastHead(False, enc_in, d_model * patch_num, pred_len, head_dropout)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        z = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        z = self.pad(z).unfold(-1, self.patch_len, self.stride)  # [B, C, N, patch_len]
        tokens = z.reshape(batch * self.enc_in, z.shape[2], self.patch_len)
        h = self.embed_drop(self.extractors(tokens) + self.position)
        for block in self.blocks:
            h = block(h)
        h = h.reshape(batch, self.enc_in, h.shape[1], h.shape[2]).transpose(2, 3)  # [B, C, d, N]
        return self.revin(self.head(h).transpose(1, 2), "denorm")

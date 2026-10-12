"""Independent DTAF implementation derived from its published method.

Temporal Stabilizing Fusion removes sample-dependent nuisance patterns before
causal aggregation. Frequency Wave Modeling selects bins with strong adjacent
spectral change. No source from the reference-only repository is used here.
"""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.instance_norm import destandardize, standardize


def patchify(values: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """Repeat-pad and return ``(batch*channels, patches, patch_len)``."""
    padded = torch.cat((values, values[:, -1:].expand(-1, stride, -1)), dim=1)
    return padded.transpose(1, 2).unfold(-1, patch_len, stride).flatten(0, 1)


class TemporalStabilizingFusion(nn.Module):
    """TFS nuisance-MoE subtraction followed by history/current fusion."""
    def __init__(self, width: int, experts: int, hidden: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))
            for _ in range(experts)
        ])
        self.router = nn.Linear(width, experts)
        self.history_score = nn.Linear(width, 1)
        self.current_gate = nn.Linear(width, width)
        self.history_value = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = self.router(tokens).softmax(-1)
        nuisance = torch.stack([expert(tokens) for expert in self.experts], -2)
        stable = tokens - torch.sum(weights.unsqueeze(-1) * nuisance, dim=-2)
        length = tokens.shape[1]
        scores = self.history_score(stable).squeeze(-1).unsqueeze(1).expand(-1, length, -1)
        causal = torch.tril(torch.ones(length, length, device=tokens.device, dtype=torch.bool))
        history = scores.masked_fill(~causal, -torch.inf).softmax(-1) @ self.history_value(stable)
        current = torch.sigmoid(self.current_gate(tokens)) * tokens
        return self.norm(tokens + self.dropout(history + current))


class FrequencyWaveModeling(nn.Module):
    """FWM emphasizes bins with the largest adjacent spectral shifts."""
    def __init__(self, width: int, top_k: int, heads: int, dropout: float):
        super().__init__()
        self.top_k = top_k
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def spectral_mask(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(tokens, dim=1)
        amplitude = spectrum.abs().mean(-1)
        difference = torch.zeros_like(amplitude)
        difference[:, 1:] = (amplitude[:, 1:] - amplitude[:, :-1]).abs()
        k = min(self.top_k, max(1, difference.shape[1] - 1))
        indices = difference.topk(k, dim=1).indices
        mask = torch.zeros_like(difference, dtype=torch.bool).scatter(1, indices, True)
        mask[:, 0] = True
        return spectrum, mask

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        spectrum, mask = self.spectral_mask(tokens)
        waves = torch.fft.irfft(spectrum * mask.unsqueeze(-1), n=tokens.shape[1], dim=1)
        attended, _ = self.attention(waves, waves, waves, need_weights=False)
        return self.norm(waves + attended)


class DTAFBlock(nn.Module):
    def __init__(self, width, experts, hidden, top_k, heads, dropout):
        super().__init__()
        self.temporal = TemporalStabilizingFusion(width, experts, hidden, dropout)
        self.frequency = FrequencyWaveModeling(width, top_k, heads, dropout)
        self.fusion = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens):
        fused = self.fusion(torch.cat((self.temporal(tokens), self.frequency(tokens)), -1))
        return self.norm(tokens + fused)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, label_len=0, features="M",
                 d_model=32, e_layers=1, patch_len=16, stride=8, heads=2,
                 dropout=0.1, expert_num=2, expert_hidden=8, top_k=1):
        super().__init__()
        if patch_len > seq_len + stride:
            raise ValueError("invalid patch length")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        self.patch_count = 1 + (seq_len + stride - patch_len) // stride
        self.embedding = nn.Linear(patch_len, d_model)
        self.blocks = nn.ModuleList([
            DTAFBlock(d_model, expert_num, expert_hidden, top_k, heads, dropout)
            for _ in range(e_layers)
        ])
        self.head = nn.Sequential(nn.Flatten(-2), nn.Dropout(dropout),
                                  nn.Linear(self.patch_count * d_model, pred_len))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        normalized, mean, scale = standardize(x_enc, 1e-5, detach_mean=True, detach_std=True)
        tokens = self.embedding(patchify(normalized, self.patch_len, self.stride))
        for block in self.blocks:
            tokens = block(tokens)
        forecast = self.head(tokens).reshape(x_enc.shape[0], self.enc_in, self.pred_len).transpose(1, 2)
        return destandardize(forecast, mean, scale)

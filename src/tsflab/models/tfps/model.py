"""TFPS: time-frequency pattern-specific experts under patch-level distribution shift.

Independent implementation from Section 3 (Eqs. 1-12, Figs. 2-3) and Appendix L
(Algorithm 1) of Sun et al., "Learning Pattern-Specific Experts for Time Series
Forecasting Under Patch-level Distribution Shift" (arXiv 2410.09836, NeurIPS 2025),
after reading the pinned official code (``syrGitHub/TFPS`` at ``83a11827``, no license
file) to resolve omissions; nothing is copied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window

# Eq. (5): weight of the subspace-basis regularizer R, fixed in the paper.
BASIS_PENALTY = 1e-3


def patch_count(length: int, patch_len: int, stride: int) -> int:
    """Section 3.3: ``floor((L - P) / S) + 2`` patches after end padding by ``S``."""
    return (length - patch_len) // stride + 2


def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """``[B, C, L]`` -> ``[B, C, N, P]``: replicate the last value ``stride`` times and unfold."""
    return F.pad(x, (0, stride), mode="replicate").unfold(-1, patch_len, stride)


class _BatchNormTokens(nn.Module):
    """BatchNorm1d over the feature axis of ``[B, N, D]`` tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualScoreAttention(nn.Module):
    """Eq. (1) multi-head attention whose pre-softmax scores add those of the
    previous layer (the PatchTST residual-attention encoder used by the time branch)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        scores = q @ k.transpose(-1, -2) * self.head_dim**-0.5
        if prev is not None:
            scores = scores + prev
        mixed = torch.softmax(scores, dim=-1) @ v
        return self.out(mixed.transpose(1, 2).reshape(batch, tokens, -1)), scores


class TimeEncoderLayer(nn.Module):
    """Time encoder block (Fig. 2b): post-norm attention and GELU feed-forward
    sublayers with residual connections and BatchNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = ResidualScoreAttention(d_model, n_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = _BatchNormTokens(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = _BatchNormTokens(d_model)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(x, prev)
        x = self.attention_norm(x + self.attention_dropout(attended))
        return self.ff_norm(x + self.ff_dropout(self.ff(x))), scores


def fourier_mix(x: torch.Tensor) -> torch.Tensor:
    """Eq. (2): real part of the 2-D FFT over the hidden and patch axes of ``[..., N, D]``."""
    return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real


class FrequencyEncoderLayer(nn.Module):
    """Frequency encoder block (Fig. 2c): the attention sublayer is replaced by the
    Fourier sublayer of Eq. (2); post-norm LayerNorm residual sublayers."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fourier_dropout = nn.Dropout(dropout)
        self.fourier_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier_norm(x + self.fourier_dropout(fourier_mix(x)))
        return self.ff_norm(x + self.ff_dropout(self.ff(x)))


class PatternIdentifier(nn.Module):
    """Section 3.5: ``K`` subspace bases ``D = [D^(1), ..., D^(K)]``, each ``q x d``
    with ``d = q // K``, Gaussian-initialised (Algorithm 1)."""

    def __init__(self, width: int, n_clusters: int, eta: float) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.eta = eta
        self.d = width // n_clusters
        self.bases = nn.Parameter(torch.randn(width, n_clusters * self.d))

    def affinity(self, z: torch.Tensor) -> torch.Tensor:
        """Eq. (6): ``s_ij`` proportional to ``||z_i^T D^(j)||^2 + eta d`` over ``[M, q]`` patches."""
        projected = (z @ self.bases).view(z.shape[0], self.n_clusters, self.d)
        energy = projected.pow(2).sum(-1) + self.eta * self.d
        return energy / energy.sum(-1, keepdim=True)

    def penalty(self) -> torch.Tensor:
        """Eqs. (3)-(5) as in the official code: ``1e-3 (||D^T D * I - I||_F + ||D^T D * O||_F)``
        with ``O`` zero on the diagonal ``d x d`` blocks."""
        gram = self.bases.T @ self.bases
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        blocks = torch.block_diag(*[torch.ones(self.d, self.d, device=gram.device, dtype=gram.dtype)] * self.n_clusters)
        return BASIS_PENALTY * (torch.linalg.norm(gram * eye - eye) + torch.linalg.norm(gram * (1.0 - blocks)))


def refined_affinity(s: torch.Tensor) -> torch.Tensor:
    """Eq. (7): ``s_ij^2 / sum_i s_ij`` renormalized over ``j`` (a fixed, gradient-free target)."""
    weight = s.pow(2) / s.sum(0, keepdim=True)
    return weight / weight.sum(1, keepdim=True)


def clustering_loss(identifier: PatternIdentifier, s: torch.Tensor, beta: float) -> torch.Tensor:
    """Eq. (9) ``L_PI = R + beta * KL(S_hat || S)``; the KL is averaged over its entries
    (official ``kl_div`` default) and the target is detached."""
    target = refined_affinity(s.detach())
    kl = F.kl_div(s.clamp_min(1e-12).log(), target, reduction="mean")
    return identifier.penalty() + beta * kl


class PatternExperts(nn.Module):
    """Section 3.6 MoPE: ``G(s) = Softmax(TopK(s))`` (Eq. 10) weights two-layer ReLU
    MLP experts on each patch feature (Eq. 11)."""

    def __init__(self, width: int, n_experts: int, top_k: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width), nn.Dropout(dropout))
            for _ in range(n_experts)
        )

    def gate(self, s: torch.Tensor) -> torch.Tensor:
        values, index = s.topk(self.top_k, dim=-1)
        logits = torch.full_like(s, float("-inf")).scatter(-1, index, values)
        return torch.softmax(logits, dim=-1)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        weights = self.gate(s)  # [M, K]; zero outside each patch's top-k experts
        out = torch.zeros_like(z)
        for k, expert in enumerate(self.experts):
            index = (weights[:, k] > 0).nonzero(as_tuple=True)[0]
            if index.numel():
                out = out.index_add(0, index, expert(z[index]) * weights[index, k : k + 1])
        return out


class Model(nn.Module):
    """TFPS forecaster for ``[B, seq_len, enc_in]`` histories."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        e_layers: int = 3,
        n_heads: int = 4,
        d_model: int = 16,
        d_ff: int = 128,
        dropout: float = 0.3,
        head_dropout: float = 0.0,
        t_experts: int = 4,
        t_top_k: int = 1,
        f_experts: int = 4,
        f_top_k: int = 1,
        eta: float = 5.0,
        beta: float = 0.1,
        lambda_time: float = 1.0,
        lambda_freq: float = 1.0,
        individual: bool = True,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.beta = beta
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq
        self.patches_in = patch_count(seq_len, patch_len, stride)
        self.patches_out = patch_count(pred_len, patch_len, stride)
        if self.patches_out < 1:
            raise ValueError("pred_len is too short for the patch length and stride")
        width = enc_in * d_model

        self.revin = RevIN(enc_in, affine=False)
        # Time branch (Section 3.4, Eq. 1).
        self.time_embedding = nn.Linear(patch_len, d_model)
        self.time_position = positional_encoding("zeros", True, self.patches_in, d_model)
        self.time_dropout = nn.Dropout(dropout)
        self.time_layers = nn.ModuleList(TimeEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers))
        self.time_project = nn.Linear(self.patches_in, self.patches_out)
        self.time_identifier = PatternIdentifier(width, t_experts, eta)
        self.time_experts = PatternExperts(width, t_experts, t_top_k)
        # Frequency branch (Section 3.4, Eq. 2).
        self.freq_embedding = nn.Linear(patch_len, d_model)
        self.freq_position = positional_encoding("zeros", True, self.patches_in, d_model)
        self.freq_dropout = nn.Dropout(dropout)
        self.freq_layers = nn.ModuleList(FrequencyEncoderLayer(d_model, d_ff, dropout) for _ in range(e_layers))
        self.freq_project = nn.Linear(self.patches_in, self.patches_out)
        self.freq_identifier = PatternIdentifier(width, f_experts, eta)
        self.freq_experts = PatternExperts(width, f_experts, f_top_k)
        # Head: concat(h_t, h_f) -> per-variate flatten-linear (Algorithm 1, lines 29-31).
        self.head = FlattenForecastHead(individual, enc_in, 2 * d_model * self.patches_out, pred_len, head_dropout)

    def encode_time(self, patches: torch.Tensor) -> torch.Tensor:
        """``[B, C, N_in, P]`` -> time features ``z_t`` ``[B, C, N_in, D]``."""
        batch, channels = patches.shape[:2]
        z = self.time_dropout(self.time_embedding(patches.flatten(0, 1)) + self.time_position)
        scores = None
        for layer in self.time_layers:
            z, scores = layer(z, scores)
        return z.reshape(batch, channels, self.patches_in, -1)

    def encode_frequency(self, patches: torch.Tensor) -> torch.Tensor:
        """``[B, C, N_in, P]`` -> frequency features ``z_f`` ``[B, C, N_in, D]``."""
        batch, channels = patches.shape[:2]
        z = self.freq_dropout(self.freq_embedding(patches.flatten(0, 1)) + self.freq_position)
        for layer in self.freq_layers:
            z = layer(z)
        return z.reshape(batch, channels, self.patches_in, -1)

    @staticmethod
    def route(
        z: torch.Tensor, project: nn.Linear, identifier: PatternIdentifier, experts: PatternExperts
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``[B, C, N_in, D]`` to ``N_out`` patches, cluster every patch feature
        ``z_i in R^{C D}`` (Eq. 6) and apply the pattern experts (Eqs. 10-11).
        Returns ``(h [B, C, N_out, D], s [B * N_out, K])``."""
        batch, channels, _, d_model = z.shape
        z = project(z.transpose(-1, -2)).transpose(-1, -2)  # [B, C, N_out, D]
        tokens = z.permute(0, 2, 1, 3).reshape(-1, channels * d_model)  # [B * N_out, C * D]
        s = identifier.affinity(tokens)
        h = experts(tokens, s).reshape(batch, -1, channels, d_model).permute(0, 2, 1, 3)
        return h, s

    def forecast_with_affinity(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forecast ``[B, H, C]`` plus the time and frequency subspace affinities
        ``s_t``, ``s_f`` (``[B * N_out, K]``) used by the clustering loss."""
        check_window(x_enc, self.seq_len, self.enc_in)
        x = self.revin(x_enc, "norm")
        patches = patchify(x.transpose(1, 2), self.patch_len, self.stride)  # [B, C, N_in, P]
        h_time, s_time = self.route(
            self.encode_time(patches), self.time_project, self.time_identifier, self.time_experts
        )
        h_freq, s_freq = self.route(
            self.encode_frequency(patches), self.freq_project, self.freq_identifier, self.freq_experts
        )
        # Back to the time domain: inverse FFT over the hidden then the patch axis, real part.
        h_freq = torch.fft.ifft(torch.fft.ifft(h_freq, dim=-1), dim=-2).real
        h = torch.cat((h_time, h_freq), dim=-1).transpose(-1, -2)  # [B, C, 2D, N_out]
        return self.revin(self.head(h).transpose(1, 2), "denorm"), s_time, s_freq

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        return self.forecast_with_affinity(x_enc)[0]

    def clustering_loss(self, s_time: torch.Tensor, s_freq: torch.Tensor) -> torch.Tensor:
        """Eq. (12) auxiliary terms ``lambda_t L_PI_t + lambda_f L_PI_f``."""
        return self.lambda_time * clustering_loss(self.time_identifier, s_time, self.beta) + (
            self.lambda_freq * clustering_loss(self.freq_identifier, s_freq, self.beta)
        )

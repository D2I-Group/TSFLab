"""Local ASGMamba: multi-scale patch Mamba with adaptive spectral gating.

Paper map (Li et al., arXiv 2602.01668, Section 4 and Algorithm 1): RevIN,
per-scale patching (P in {8, 16, 32}), patch embedding plus position and node
embeddings (Eq. 3), patch-level rFFT band energies (Eq. 4) mapped to a gate
(Eq. 5), gated pre-norm residual Mamba block (Eqs. 6-7), per-scale flatten
head, softmax scale fusion (Eq. 8), and RevIN denormalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def patch_layout(seq_len: int, patch_len: int, stride: int) -> tuple[int, int]:
    """Right zero-padding and patch count for unfolding ``seq_len`` steps."""
    padded = max(seq_len, patch_len)
    padded += (stride - (padded - patch_len) % stride) % stride
    return padded - seq_len, (padded - patch_len) // stride + 1


def band_energy(patches: torch.Tensor) -> torch.Tensor:
    """Eq. (4) and band aggregation: normalized low/mid/high rFFT energy per patch.

    Bin ``j`` of a length-``P`` patch has normalized frequency ``nu_j = 2j / P``
    (fraction of Nyquist); low is ``nu <= 1/3``, mid ``1/3 < nu <= 2/3``, high
    ``nu > 2/3``. Energies are divided by the total patch energy (floor 1e-6).
    ``[..., P] -> [..., 3]``.
    """
    energy = torch.fft.rfft(patches, dim=-1).abs().pow(2)
    nu = 2.0 * torch.arange(energy.shape[-1], device=patches.device, dtype=energy.dtype) / patches.shape[-1]
    low = energy[..., nu <= 1.0 / 3.0].sum(dim=-1, keepdim=True)
    mid = energy[..., (nu > 1.0 / 3.0) & (nu <= 2.0 / 3.0)].sum(dim=-1, keepdim=True)
    high = energy[..., nu > 2.0 / 3.0].sum(dim=-1, keepdim=True)
    total = energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat((low, mid, high), dim=-1) / total


class SpectralGate(nn.Module):
    """Adaptive spectral gate: band energies -> per-token, per-feature gate ``G``.

    ``sigmoid``: Eq. (5), ``G = sigmoid(W2 ReLU(W1 v))`` in ``(0, 1)`` with a
    ``d_model // 4`` hidden layer. ``identity_centered`` (pinned official code):
    ``G = 1 + a * tanh(W2 SiLU(W1 v))`` in ``(1 - a, 1 + a)`` with a
    ``max(d_model // 4, 8)`` hidden layer and a zero-initialized ``W2`` so the
    gate starts at one.
    """

    def __init__(self, d_model: int, kind: str, amplitude: float) -> None:
        super().__init__()
        self.kind, self.amplitude = kind, amplitude
        hidden = max(d_model // 4, 8) if kind == "identity_centered" else max(d_model // 4, 1)
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        if kind == "identity_centered":
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        descriptor = band_energy(patches)
        if self.kind == "sigmoid":
            return torch.sigmoid(self.fc2(F.relu(self.fc1(descriptor))))
        return 1.0 + self.amplitude * torch.tanh(self.fc2(F.silu(self.fc1(descriptor))))


class ScaleBranch(nn.Module):
    """One patch scale: patching, embeddings, gated Mamba block, flatten head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        overlap: bool,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        gate: str,
        gate_amplitude: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = max(patch_len // 2, 1) if overlap else patch_len
        self.pad, self.num_patches = patch_layout(seq_len, patch_len, self.stride)
        self.d_model = d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)
        self.gate = SpectralGate(d_model, gate, gate_amplitude)
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model,
            expand * d_model,
            math.ceil(d_model / 16),
            d_conv,
            d_state,
            reference_dt_init=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def patches(self, series: torch.Tensor) -> torch.Tensor:
        """``[B, C, L] -> [B, C, N_k, P_k]`` with right zero-padding."""
        return F.pad(series, (0, self.pad)).unfold(-1, self.patch_len, self.stride)

    def forward(self, series: torch.Tensor, node_embedding: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = series.shape
        patches = self.patches(series)
        # Eq. (3): Z0 = PatchEmb(X_p) + E_pos + E_node.
        tokens = self.patch_embedding(patches) + self.position + node_embedding
        tokens = tokens.reshape(batch * channels, self.num_patches, self.d_model)
        gate = self.gate(patches.reshape(batch * channels, self.num_patches, self.patch_len))
        # Eq. (6) and block output: Z_out = Dropout(Mamba(LN(Z) * G)) + Z.
        encoded = self.dropout(self.mamba(self.norm(tokens) * gate)) + tokens
        return self.head(encoded.reshape(batch, channels, -1))


class Model(nn.Module):
    """ASGMamba: RevIN, three gated patch-Mamba branches, softmax scale fusion."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_sizes: tuple[int, ...] = (8, 16, 32),
        overlap: bool = False,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        gate: str = "identity_centered",
        gate_amplitude: float = 0.5,
        revin_affine: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_sizes = tuple(patch_sizes)
        self.revin = RevIN(enc_in, eps=1e-5, affine=revin_affine)
        self.node_embedding = nn.Parameter(torch.randn(1, enc_in, 1, d_model) * 0.02)
        self.branches = nn.ModuleList(
            ScaleBranch(
                seq_len,
                pred_len,
                patch_len,
                overlap,
                d_model,
                d_state,
                d_conv,
                expand,
                dropout,
                gate,
                gate_amplitude,
            )
            for patch_len in self.patch_sizes
        )
        self.scale_logits = nn.Parameter(torch.ones(len(self.patch_sizes)))

    def scale_weights(self) -> torch.Tensor:
        """Eq. (8): softmax over the learnable global scale logits."""
        return torch.softmax(self.scale_logits, dim=0)

    def branch_forecasts(self, x_enc: torch.Tensor) -> list[torch.Tensor]:
        """Per-scale forecasts in normalized space, each ``[B, C, H]``."""
        series = self.revin(x_enc, "norm").transpose(1, 2)
        return [branch(series, self.node_embedding) for branch in self.branches]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        forecasts = torch.stack(self.branch_forecasts(x_enc), dim=0)
        fused = (self.scale_weights().view(-1, 1, 1, 1) * forecasts).sum(dim=0)
        return self.revin(fused.transpose(1, 2), "denorm")

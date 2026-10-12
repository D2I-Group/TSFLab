"""Independent Sonnet implementation from its published spectral equations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableWavelets(nn.Module):
    def __init__(self, length: int, width: int, atoms: int) -> None:
        super().__init__()
        self.length = length
        self.alpha = nn.Parameter(torch.randn(atoms, width) * 0.02)
        self.beta = nn.Parameter(torch.randn(atoms, width) * 0.02)
        self.gamma = nn.Parameter(torch.randn(atoms, width) * 0.02)

    def atoms(self) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, self.length, device=self.alpha.device, dtype=self.alpha.dtype)
        t = t.reshape(1, self.length, 1)
        alpha = F.softplus(self.alpha).unsqueeze(1)
        beta = self.beta.unsqueeze(1)
        gamma = self.gamma.unsqueeze(1)
        return torch.exp(-alpha * t.square()) * torch.cos(beta * t + gamma * t.square())

    def forward(self, embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.atoms()
        return embedded.unsqueeze(0) * atoms.unsqueeze(1), atoms


class SpectralCoherence(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.output = nn.Linear(width, width)

    def forward(self, wavelet_space: torch.Tensor) -> torch.Tensor:
        query = torch.fft.rfft(self.query(wavelet_space), dim=-1)
        key = torch.fft.rfft(self.key(wavelet_space), dim=-1)
        value = self.value(wavelet_space)
        cross = (query * key.conj()).mean(dim=-1)
        query_power = (query * query.conj()).real.mean(dim=-1)
        key_power = (key * key.conj()).real.mean(dim=-1)
        coherence = cross.abs().square() / (query_power * key_power).clamp_min(1e-6)
        attention = self.dropout((coherence / wavelet_space.shape[-1] ** 0.5).softmax(dim=-1))
        attended = attention.unsqueeze(-1) * value
        return self.output(attended + self.mlp(attended))


class StableKoopman(nn.Module):
    def __init__(self, atoms: int) -> None:
        super().__init__()
        self.real = nn.Parameter(torch.eye(atoms) + torch.randn(atoms, atoms) * 0.01)
        self.imag = nn.Parameter(torch.randn(atoms, atoms) * 0.01)
        self.phase = nn.Parameter(torch.randn(atoms) * 0.02)

    def operator(self) -> torch.Tensor:
        unitary, _ = torch.linalg.qr(torch.complex(self.real, self.imag))
        diagonal = torch.diag(torch.polar(torch.ones_like(self.phase), self.phase))
        return unitary @ diagonal @ unitary.mH

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        complex_x = torch.complex(x, torch.zeros_like(x))
        return torch.einsum("kq,qbld->kbld", self.operator(), complex_x)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 16,
        num_wavelets: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.embedding = nn.Linear(enc_in, d_model)
        self.wavelets = LearnableWavelets(seq_len, d_model, num_wavelets)
        self.coherence = SpectralCoherence(d_model, dropout)
        self.koopman = StableKoopman(num_wavelets)
        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, enc_in, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        embedded = self.embedding(x_enc)
        wavelet_space, atoms = self.wavelets(embedded)
        attended = self.coherence(wavelet_space)
        evolved = self.koopman(attended).real
        reconstructed = (evolved * atoms.unsqueeze(1)).sum(dim=0)
        decoded = self.decoder(reconstructed.transpose(1, 2))
        output = F.adaptive_avg_pool1d(decoded, self.pred_len).transpose(1, 2)
        return output

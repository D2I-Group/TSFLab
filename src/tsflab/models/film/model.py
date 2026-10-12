"""Clean-room FiLM implementation from its Legendre--Fourier formulation."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


def legt_transition(order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous translated-Legendre state matrices used by the LPU."""
    degree = torch.arange(order, dtype=torch.float64)
    row = degree[:, None]
    column = degree[None, :]
    scale = 2.0 * row + 1.0
    signs = torch.where(
        row < column,
        -torch.ones_like(row + column),
        torch.pow(-torch.ones_like(row + column), row - column + 1.0),
    )
    matrix = signs * scale
    input_vector = torch.pow(-torch.ones_like(row), row) * scale
    return matrix, input_vector.squeeze(1)


def bilinear_discretize(
    matrix: torch.Tensor, input_vector: torch.Tensor, step: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear discretization of ``c'=Ac+Bx``."""
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
    left = identity - 0.5 * step * matrix
    discrete_matrix = torch.linalg.solve(left, identity + 0.5 * step * matrix)
    discrete_input = torch.linalg.solve(left, step * input_vector)
    return discrete_matrix.float(), discrete_input.float()


def legendre_basis(length: int, order: int) -> torch.Tensor:
    """Evaluate P_0...P_(N-1) on the normalized forecast grid."""
    points = 1.0 - 2.0 * (torch.arange(length, dtype=torch.float32) + 0.5) / length
    basis = [torch.ones_like(points)]
    if order > 1:
        basis.append(points)
    for degree in range(2, order):
        basis.append(
            ((2 * degree - 1) * points * basis[-1] - (degree - 1) * basis[-2])
            / degree
        )
    return torch.stack(basis, dim=-1)


class LegendreProjection(nn.Module):
    """LPU recurrence ``C_t=A C_(t-1)+B x_t`` from Section 3.1."""

    def __init__(self, order: int, input_length: int, output_length: int) -> None:
        super().__init__()
        continuous_a, continuous_b = legt_transition(order)
        discrete_a, discrete_b = bilinear_discretize(
            continuous_a, continuous_b, 1.0 / input_length
        )
        self.register_buffer("transition", discrete_a)
        self.register_buffer("input_vector", discrete_b)
        self.register_buffer("reconstruction", legendre_basis(output_length, order))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        state = values.new_zeros(batch, channels, self.transition.shape[0])
        trajectory = []
        for step in range(length):
            state = torch.einsum("bcn,mn->bcm", state, self.transition)
            state = state + values[:, step].unsqueeze(-1) * self.input_vector
            trajectory.append(state)
        return torch.stack(trajectory, dim=2)

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcn,tn->btc", coefficients, self.reconstruction)


class LowRankFourierLayer(nn.Module):
    """Lowest-mode Fourier selection with complex low-rank weights."""

    def __init__(self, length: int, order: int, rank: int, ratio: float) -> None:
        super().__init__()
        self.length = length
        self.modes = max(1, min(length // 2 + 1, math_ceil((length // 2 + 1) * ratio)))
        scale = (order * rank) ** -0.5
        self.left_real = nn.Parameter(torch.randn(self.modes, order, rank) * scale)
        self.left_imag = nn.Parameter(torch.randn(self.modes, order, rank) * scale)
        self.right_real = nn.Parameter(torch.randn(self.modes, rank, order) * scale)
        self.right_imag = nn.Parameter(torch.randn(self.modes, rank, order) * scale)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(trajectory, dim=2, norm="ortho")
        selected = spectrum[:, :, : self.modes]
        left = torch.complex(self.left_real, self.left_imag)
        right = torch.complex(self.right_real, self.right_imag)
        compressed = torch.einsum("bcfn,fnr->bcfr", selected, left)
        filtered = torch.einsum("bcfr,frn->bcfn", compressed, right)
        output = torch.zeros_like(spectrum)
        output[:, :, : self.modes] = filtered
        return torch.fft.irfft(output, n=self.length, dim=2, norm="ortho")


def math_ceil(value: float) -> int:
    """Small local integer ceiling without a NumPy/SciPy dependency."""
    integer = int(value)
    return integer if value == integer else integer + 1


class FiLMExpert(nn.Module):
    def __init__(
        self, input_length: int, pred_len: int, order: int, rank: int, ratio: float
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.projection = LegendreProjection(order, input_length, pred_len)
        self.frequency_layer = LowRankFourierLayer(input_length, order, rank, ratio)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        history = values[:, -self.input_length :]
        trajectory = self.projection(history)
        filtered = self.frequency_layer(trajectory)
        return self.projection.reconstruct(filtered[:, :, -1])


class Model(nn.Module):
    """FiLM with LPU, low-rank FEL, and mixture of multiscale experts."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        label_len: int = 0,
        features: str = "M",
        ratio: float = 0.5,
        multiscale: tuple[int, ...] = (1, 2, 4),
        order: int = 64,
        rank: int = 4,
    ) -> None:
        super().__init__()
        del label_len, features
        if min(seq_len, pred_len) < 1:
            raise ValueError("FiLM dimensions must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.revin = RevIN(enc_in)
        lengths = [min(seq_len, scale * pred_len) for scale in multiscale]
        self.experts = nn.ModuleList(
            [FiLMExpert(length, pred_len, order, rank, ratio) for length in lengths]
        )
        self.expert_mixture = nn.Linear(len(lengths), 1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(
                f"x_enc must have shape (batch, {self.seq_len}, {self.channels})"
            )
        normalized = self.revin(x_enc, "norm")
        forecasts = torch.stack([expert(normalized) for expert in self.experts], dim=-1)
        forecast = self.expert_mixture(forecasts).squeeze(-1)
        return self.revin(forecast, "denorm")

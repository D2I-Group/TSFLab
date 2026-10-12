"""Clean-room Sumba from the NeurIPS 2024 structured-basis formulation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.instance_norm import destandardize, standardize


class StructuredMatrixBasis(nn.Module):
    """Low-rank, row-stochastic basis matrices with convex dynamic mixing."""
    def __init__(self, nodes: int, features: int, basis_count: int, basis_rank: int) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.randn(basis_count, nodes, basis_rank) * 0.1)
        self.right = nn.Parameter(torch.randn(basis_count, nodes, basis_rank) * 0.1)
        self.spectrum = nn.Parameter(torch.ones(basis_count, basis_rank))
        self.coefficients = nn.Sequential(nn.Linear(features, features), nn.GELU(), nn.Linear(features, basis_count))

    def matrices(self) -> torch.Tensor:
        logits = torch.einsum("mnr,mr,mkr->mnk", self.left, self.spectrum, self.right)
        return logits.softmax(dim=-1)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.coefficients(context.mean(dim=(1, 2))).softmax(dim=-1)
        graph = torch.einsum("bm,mnk->bnk", weights, self.matrices())
        return graph, weights

    def diversity_penalty(self) -> torch.Tensor:
        flat = F.normalize(self.matrices().flatten(1), dim=-1)
        gram = flat @ flat.transpose(0, 1)
        return (gram - torch.eye(gram.size(0), device=gram.device)).square().mean()


class MultiScaleTemporalConv(nn.Module):
    """Parallel causal temporal filters for multi-scale temporal dependence."""
    def __init__(self, features: int, kernels: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.kernels = kernels
        self.filters = nn.ModuleList(nn.Conv2d(features, features, (1, kernel), groups=features) for kernel in kernels)
        self.gates = nn.ModuleList(nn.Conv2d(features, features, (1, kernel), groups=features) for kernel in kernels)
        self.combine = nn.Conv2d(features * len(kernels), features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [B,T,N,D] -> [B,D,N,T]
        x = values.permute(0, 3, 2, 1)
        branches = []
        for kernel, filt, gate in zip(self.kernels, self.filters, self.gates):
            padded = F.pad(x, (kernel - 1, 0, 0, 0))
            branches.append(torch.tanh(filt(padded)) * torch.sigmoid(gate(padded)))
        return self.dropout(self.combine(torch.cat(branches, dim=1))).permute(0, 3, 2, 1)


class DynamicBasisGraphConv(nn.Module):
    """Diffuse node features over a convex mixture of structured bases."""
    def __init__(self, features: int, steps: int, mix: float, dropout: float) -> None:
        super().__init__()
        self.steps, self.mix = steps, float(mix)
        self.project = nn.Linear(features * (steps + 1), features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        states, current = [values], values
        for _ in range(self.steps):
            propagated = torch.einsum("bnm,btmd->btnd", graph, current)
            current = self.mix * values + (1.0 - self.mix) * propagated
            states.append(current)
        return self.dropout(self.project(torch.cat(states, dim=-1)))


class SumbaBlock(nn.Module):
    def __init__(self, nodes: int, features: int, basis_count: int, basis_rank: int,
                 kernels: tuple[int, ...], diffusion_steps: int, mix: float, dropout: float) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(features)
        self.temporal = MultiScaleTemporalConv(features, kernels, dropout)
        self.norm_g = nn.LayerNorm(features)
        self.basis = StructuredMatrixBasis(nodes, features, basis_count, basis_rank)
        self.graph = DynamicBasisGraphConv(features, diffusion_steps, mix, dropout)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = values + self.temporal(self.norm_t(values))
        graph, weights = self.basis(values)
        return values + self.graph(self.norm_g(values), graph), weights


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, label_len: int = 0,
                 features: str = "M", d_model: int = 32, basis_count: int = 4,
                 basis_rank: int = 8, temporal_kernels: tuple[int, ...] = (2, 3, 5),
                 depth: int = 2, diffusion_steps: int = 2, mix: float = 0.1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("Sumba dimensions must be positive")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.input_projection = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(SumbaBlock(enc_in, d_model, basis_count, basis_rank, temporal_kernels, diffusion_steps, mix, dropout) for _ in range(depth))
        self.temporal_head = nn.Linear(seq_len * d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f"Sumba expects [B, {self.seq_len}, C]")
        normalized, mean, scale = standardize(x_enc, detach_std=True)
        hidden = self.input_projection(normalized.unsqueeze(-1))
        for block in self.blocks:
            hidden, _ = block(hidden)
        forecast = self.temporal_head(hidden.permute(0, 2, 1, 3).flatten(-2)).transpose(1, 2)
        return destandardize(forecast, mean, scale)

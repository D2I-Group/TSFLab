"""SDGF network: static and multi-scale dynamic graph fusion for forecasting.

Independent implementation of Wang et al., "SDGF: Fusing Static and Multi-Scale
Dynamic Correlations for Multivariate Time Series Forecasting" (arXiv
2509.18135), checked against shaoxun6033/SDGFNet@bf76b56f (Apache-2.0).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN

StaticGraph = Literal["rbf", "pcc"]

# Daubechies-4 (8-tap) analysis filters in the PyWavelets convention.
DB4_DEC_LO = (
    -0.010597401785069032,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
    0.6308807679295904,
    0.7148465705525415,
    0.23037781330885523,
)
DB4_DEC_HI = tuple(((-1) ** (k + 1)) * DB4_DEC_LO[len(DB4_DEC_LO) - 1 - k] for k in range(len(DB4_DEC_LO)))


def _symmetric_index(index: torch.Tensor, length: int) -> torch.Tensor:
    """Half-sample symmetric extension: ``x[-1] = x[0]``, ``x[n] = x[n - 1]``, repeated."""
    period = 2 * length
    m = torch.remainder(index, period)
    return torch.where(m < length, m, period - 1 - m)


def _analysis_matrices(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One DWT level as matrices ``[M, length]`` with ``M = floor((length + 7) / 2)``.

    ``c[o] = sum_j h[j] x_ext[2o + 1 - j]`` on the symmetric extension, the
    PyWavelets ``mode="symmetric"`` alignment.
    """
    taps = len(DB4_DEC_LO)
    out = (length + taps - 1) // 2
    o = torch.arange(out).view(-1, 1)
    j = torch.arange(taps).view(1, -1)
    source = _symmetric_index(2 * o + 1 - j, length)  # [M, F]
    lo = torch.zeros(out, length, dtype=torch.float64)
    hi = torch.zeros(out, length, dtype=torch.float64)
    rows = o.expand(-1, taps)
    lo.index_put_((rows, source), torch.tensor(DB4_DEC_LO, dtype=torch.float64).expand(out, -1), accumulate=True)
    hi.index_put_((rows, source), torch.tensor(DB4_DEC_HI, dtype=torch.float64).expand(out, -1), accumulate=True)
    return lo, hi


def _synthesis_matrices(length: int, coefficients: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse level ``[length, M]``: ``y[t] = sum_o c[o] h[2o + 1 - t]`` for ``0 <= t < length``.

    The transposed-filter synthesis of the extended analysis, cropped to the
    original samples, inverts :func:`_analysis_matrices` exactly.
    """
    taps = len(DB4_DEC_LO)
    t = torch.arange(length).view(-1, 1)
    o = torch.arange(coefficients).view(1, -1)
    k = 2 * o + 1 - t
    valid = (k >= 0) & (k < taps)
    k = k.clamp(0, taps - 1)
    lo = torch.where(valid, torch.tensor(DB4_DEC_LO, dtype=torch.float64)[k], torch.zeros((), dtype=torch.float64))
    hi = torch.where(valid, torch.tensor(DB4_DEC_HI, dtype=torch.float64)[k], torch.zeros((), dtype=torch.float64))
    return lo, hi


def wavelet_band_projections(length: int, level: int) -> torch.Tensor:
    """``[level + 1, length, length]`` linear maps from a series to its reconstructed db4 bands.

    Band 0 keeps only the level-``level`` approximation, band ``j`` (1-based) only
    the level-``j`` detail (finest first); every band is inverted back to
    ``length`` samples and the bands sum to the identity.
    """
    lengths = [length]
    analysis, synthesis = [], []
    for _ in range(level):
        lo, hi = _analysis_matrices(lengths[-1])
        analysis.append((lo, hi))
        synthesis.append(_synthesis_matrices(lengths[-1], lo.shape[0]))
        lengths.append(lo.shape[0])

    def lift(matrix: torch.Tensor, upto: int) -> torch.Tensor:
        # Inverse low-pass from level `upto` down to the original length.
        for j in range(upto - 1, -1, -1):
            matrix = synthesis[j][0] @ matrix
        return matrix

    approx = torch.eye(length, dtype=torch.float64)
    bands = []
    details = []
    for j in range(level):
        details.append(lift(synthesis[j][1] @ analysis[j][1] @ approx, j))
        approx = analysis[j][0] @ approx
    bands.append(lift(approx, level))
    bands.extend(details)
    return torch.stack(bands)


class StaticMixProp(nn.Module):
    """Static graph propagation (Eq. 3): ``h <- a x + (1 - a) A h`` for ``depth`` hops, then a 1x1 conv over all hops."""

    def __init__(self, channels: int, depth: int, alpha: float) -> None:
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.mlp = nn.Conv2d((depth + 1) * channels, channels, kernel_size=1)

    @staticmethod
    def normalize(adj: torch.Tensor) -> torch.Tensor:
        """Add self loops and divide row ``v`` by the column sum of node ``v`` (official convention)."""
        adj = adj + torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device)
        return adj / adj.sum(dim=-2).unsqueeze(-1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a = self.normalize(adj)  # [B, N, N]
        h, hops = x, [x]
        for _ in range(self.depth):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum("bcwl,bvw->bcvl", h, a)
            hops.append(h)
        return self.mlp(torch.cat(hops, dim=1))


class DynamicMixProp(nn.Module):
    """Dynamic graph propagation (Eqs. 4-5) with a per-time-step adjacency from ``tanh`` node embeddings."""

    def __init__(self, channels: int, depth: int, alpha: float) -> None:
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.source = nn.Conv2d(channels, channels, kernel_size=1)
        self.target = nn.Conv2d(channels, channels, kernel_size=1)
        self.mlp = nn.Conv2d((depth + 1) * channels, channels, kernel_size=1)

    def adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, N, N, L]``: mean of the row- and column-softmax of ``tanh(W1 x)^T tanh(W2 x)`` at each step."""
        scores = torch.einsum("bcvl,bcwl->bvwl", torch.tanh(self.source(x)), torch.tanh(self.target(x)))
        rows = torch.softmax(scores, dim=2)
        cols = torch.softmax(scores.transpose(1, 2), dim=2)
        return (rows + cols) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = self.adjacency(x)
        h, hops = x, [x]
        for _ in range(self.depth):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum("bcvl,bvwl->bcwl", h, adj)
            hops.append(h)
        return self.mlp(torch.cat(hops, dim=1))


class AttentionGatedFusion(nn.Module):
    """Eqs. (7)-(9): score every graph output with a learned query and return their softmax-weighted sum."""

    def __init__(self, seq_len: int, num_nodes: int, dim: int) -> None:
        super().__init__()
        self.summary = nn.Linear(seq_len * num_nodes, dim)
        self.norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, dim)
        self.query = nn.Parameter(torch.randn(1, dim))

    def weights(self, stacked: torch.Tensor) -> torch.Tensor:
        """``[B, G]`` attention weights for ``[B, G, L, N]`` graph outputs."""
        keys = self.key(self.norm(self.summary(stacked.flatten(2))))
        return torch.softmax(keys @ self.query.squeeze(0), dim=-1)

    def forward(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(outputs, dim=1)
        return (stacked * self.weights(stacked)[..., None, None]).sum(dim=1)


class InceptionBlock(nn.Module):
    """Eqs. (10)-(12): 1x1 lift, four dilated convolutions (k in {3, 5}, d in {1, 2}), bottleneck, residual, LayerNorm.

    As in the official code, the ``L`` history positions are the convolution
    channels and the kernels slide over the variable axis.
    """

    def __init__(self, seq_len: int, width: int) -> None:
        super().__init__()
        self.lift = nn.Conv1d(seq_len, width, kernel_size=1)
        self.branches = nn.ModuleList(
            nn.Conv1d(width, seq_len, kernel_size=k, dilation=d, padding="same")
            for k, d in ((3, 1), (3, 2), (5, 1), (5, 2))
        )
        self.bottleneck = nn.Conv1d(4 * seq_len, seq_len, kernel_size=1)
        self.residual = nn.Conv1d(width, seq_len, kernel_size=1)
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lifted = self.lift(x)  # [B, width, N]
        mixed = self.bottleneck(torch.cat([branch(lifted) for branch in self.branches], dim=1))
        return self.norm((mixed + self.residual(lifted)).transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    """SDGF forecaster with the four-input TSFLab interface."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        decomp_level: int = 3,
        conv_channel: int = 32,
        gcn_depth: int = 2,
        propalpha: float = 0.3,
        static_graph: StaticGraph = "rbf",
        rbf_sigma: float = 1.0,
        fusion_dim: int = 128,
        inception_width: int = 512,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.static_graph = static_graph
        self.rbf_sigma = rbf_sigma
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)

        self.static_start = nn.Conv2d(1, conv_channel, kernel_size=1)
        self.static_gcn = StaticMixProp(conv_channel, gcn_depth, propalpha)
        self.static_end = nn.Conv2d(conv_channel, seq_len, kernel_size=(1, seq_len))
        self.static_norm = nn.LayerNorm(seq_len)

        self.register_buffer(
            "band_projections", wavelet_band_projections(seq_len, decomp_level).float(), persistent=False
        )
        self.dynamic_start = nn.Conv2d(1, conv_channel, kernel_size=1)
        self.dynamic_gcn = DynamicMixProp(conv_channel, gcn_depth, propalpha)
        self.dynamic_end = nn.Conv2d(conv_channel, seq_len, kernel_size=(1, seq_len))

        self.fusion = AttentionGatedFusion(seq_len, enc_in, fusion_dim)
        self.inception = InceptionBlock(seq_len, inception_width)
        self.head = nn.Linear(seq_len, pred_len)

    def static_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample ``[B, N, N]`` prior graph from the normalized history ``[B, L, N]``.

        ``rbf``: ``exp(-||x_n - x_m||^2 / (2 sigma^2))`` (official code);
        ``pcc``: ``softmax(relu(pearson))`` over each row (Eqs. 1-2).
        """
        series = x.transpose(1, 2)  # [B, N, L]
        if self.static_graph == "rbf":
            dist = torch.cdist(series, series).pow(2)
            return torch.exp(-dist / (2 * self.rbf_sigma**2))
        centred = series - series.mean(dim=-1, keepdim=True)
        cov = centred @ centred.transpose(1, 2)
        std = centred.pow(2).sum(dim=-1).sqrt()
        pcc = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-9)
        return torch.softmax(F.relu(pcc), dim=-1)

    def static_branch(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L, N] -> [B, L, N]``: GCN over the prior graph, full-history conv, residual LayerNorm over time."""
        nodes = x.transpose(1, 2)  # [B, N, L]
        h = self.static_start(nodes.unsqueeze(1))
        h = F.gelu(self.static_gcn(h, self.static_adjacency(x)))
        h = self.static_end(h).squeeze(-1).transpose(1, 2)  # [B, N, L]
        return self.static_norm(nodes + h).transpose(1, 2)

    def wavelet_bands(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-level db4 decomposition: ``[B, L, N] -> [level + 1, B, L, N]`` bands summing to ``x``."""
        return torch.einsum("kts,bsn->kbtn", self.band_projections.to(x.dtype), x)

    def dynamic_branch(self, band: torch.Tensor) -> torch.Tensor:
        """``[B, L, N] -> [B, L, N]``: dynamic-graph GCN and full-history conv shared by every band."""
        h = self.dynamic_start(band.transpose(1, 2).unsqueeze(1))
        h = self.dynamic_gcn(h)
        return self.dynamic_end(h).squeeze(-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.revin(x_enc, "norm")
        outputs = [self.static_branch(x)] + [self.dynamic_branch(band) for band in self.wavelet_bands(x)]
        fused = self.fusion(outputs)  # [B, L, N]
        features = self.inception(fused)  # [B, L, N]
        forecast = self.head(features.transpose(1, 2)).transpose(1, 2)
        return self.revin(forecast, "denorm")

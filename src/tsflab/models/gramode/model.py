"""GRAM-ODE: graph-based multi-ODE network for traffic forecasting (Liu et al., TMLR 2023).

Two graphs drive two streams: the road connection map and a DTW graph built from
daily node profiles of the training series (Eqs. 1-3). Each stream has parallel
channels of stacked GRAM-ODE layers. A layer is TCN -> multi ODE-GNN block -> TCN
-> BatchNorm over nodes (Sec. 4.2, Algorithm 1). The block integrates three
graph ODEs with explicit Euler steps: a global node ODE over the whole window, an
edge ODE over pairwise node features, and local node ODEs started from attention
tokens. Local messages are clipped to a learnable band around the global message
(Eq. 18), the three messages are fused by pairwise sigmoid gating (Eq. 19), and a
gated residual updates the input (Eq. 20). A multi-head attention across nodes over
the concatenated channel outputs emits the horizon (Sec. 4.3).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.adj_norm import symmetric_normalized_laplacian
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


# --------------------------------------------------------------------- graphs
def normalized_support(adj, alpha: float = 0.8) -> np.ndarray:
    """Eq. (3) as in the official code: ``alpha / 2 * (I + D^{-1/2} A D^{-1/2})``."""
    adj = np.asarray(adj, dtype=np.float64)
    identity = np.eye(adj.shape[0])
    return alpha / 2.0 * (2.0 * identity - symmetric_normalized_laplacian(adj))


def daily_profiles(series: torch.Tensor, steps_per_day: int) -> torch.Tensor:
    """``[T, N]`` -> ``[N, S]`` mean day profile over the complete days of the series.

    A series shorter than one day is used as a single profile.
    """
    days = series.shape[0] // steps_per_day
    if days == 0:
        return series.transpose(0, 1)
    whole = series[: days * steps_per_day].reshape(days, steps_per_day, series.shape[1])
    return whole.mean(dim=0).transpose(0, 1)


def dtw_distances(profiles: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """Exact DTW distance (absolute-difference cost) between every pair of rows.

    ``profiles`` is ``[N, S]``; returns a symmetric ``[N, N]`` matrix with a zero
    diagonal. The dynamic program runs over anti-diagonals, vectorized over pairs.
    """
    profiles = profiles.to(torch.float64)
    nodes, length = profiles.shape
    rows, cols = torch.triu_indices(nodes, nodes, offset=1)
    result = profiles.new_zeros(nodes, nodes)
    index = torch.arange(length + 1)  # position i on a diagonal, i = 0..S
    for start in range(0, rows.numel(), chunk):
        a = profiles[rows[start:start + chunk]]
        b = profiles[cols[start:start + chunk]]
        pairs = a.shape[0]
        inf = torch.full((pairs, length + 1), math.inf, dtype=profiles.dtype)
        before_last, last = inf.clone(), inf.clone()
        before_last[:, 0] = 0.0  # D[0, 0] on diagonal k = 0; diagonal k = 1 is all inf
        for k in range(2, 2 * length + 1):
            j = k - index  # column paired with row i on this diagonal
            valid = (index >= 1) & (index <= length) & (j >= 1) & (j <= length)
            cost = (a[:, (index - 1).clamp(0, length - 1)] - b[:, (j - 1).clamp(0, length - 1)]).abs()
            up = torch.cat([inf[:, :1], last[:, :-1]], dim=1)  # D[i-1, j]
            diagonal = torch.cat([inf[:, :1], before_last[:, :-1]], dim=1)  # D[i-1, j-1]
            best = torch.minimum(torch.minimum(up, last), diagonal)  # last[i] is D[i, j-1]
            current = torch.where(valid, cost + best, inf)
            before_last, last = last, current
        result[rows[start:start + chunk], cols[start:start + chunk]] = last[:, length]
    return result + result.transpose(0, 1)


def semantic_adjacency(distances: torch.Tensor, sigma: float = 0.1, threshold: float = 0.6) -> torch.Tensor:
    """Official DTW graph: z-score all distances, Gaussian kernel, keep entries above ``threshold``."""
    z = (distances - distances.mean()) / distances.std(correction=0)
    return (torch.exp(-z.square() / sigma**2) > threshold).to(distances.dtype)


# ----------------------------------------------------------------- layers
class TemporalConvNet(nn.Module):
    """Causal dilated ``(1, k)`` convolutions with ReLU and dropout plus a residual path.

    Input and output are ``[B, N, T, C]``; level ``i`` has dilation ``2^i``.
    """

    def __init__(self, in_dim: int, channels: tuple[int, ...], kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.paddings = [(kernel_size - 1) * 2**level for level in range(len(channels))]
        self.convs = nn.ModuleList()
        width = in_dim
        for level, out_dim in enumerate(channels):
            conv = nn.Conv2d(width, out_dim, (1, kernel_size), dilation=(1, 2**level))
            nn.init.normal_(conv.weight, 0.0, 0.01)
            self.convs.append(conv)
            width = out_dim
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv2d(in_dim, channels[-1], (1, 1)) if in_dim != channels[-1] else None
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 3, 1, 2)  # [B, C, N, T]
        out = y
        for conv, padding in zip(self.convs, self.paddings):
            out = self.dropout(F.relu(conv(F.pad(out, (padding, 0)))))
        residual = y if self.downsample is None else self.downsample(y)
        return F.relu(out + residual).permute(0, 2, 3, 1)


class TemporalGate(nn.Module):
    """Shared temporal weights ``W_s1, W_s2`` (Eqs. 6-8) and the gated temporal message."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self.w_left = nn.Parameter(torch.randn(length, length))
        self.w_right = nn.Parameter(torch.randn(length, length))

    def node(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, N, T, C]``: ``S((X_c^T W1)(X_c^T W2)^T)`` mixes time per channel (Eqs. 7-8)."""
        left = torch.einsum("bntc,to->bcon", x, self.w_left)
        right = torch.einsum("bntc,tp->bcpn", x, self.w_right)
        gate = torch.sigmoid(left @ right.transpose(-1, -2))  # [B, C, T, T]
        return torch.einsum("bcop,bnpc->bnoc", gate, x)

    def edge(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, N, N, T]``: ``X S((X W1)^T (X W2))`` per source node (Eq. 6)."""
        left = x @ self.w_left
        right = x @ self.w_right
        return x @ torch.sigmoid(left.transpose(-1, -2) @ right)


def graph_message(x: torch.Tensor, support: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(A_hat + M) x_2 H``: propagate along the node axis 1 with the shared spatial weight."""
    return torch.einsum("ij,bjlm->bilm", support + mask, x)


class NodeODE(nn.Module):
    """Global or local node ODE function (Eqs. 10 and 15, official form).

    ``f = sigmoid(a)/2 * (A + M) H - H + T(H) - H + H W' - H + H(0)`` with
    ``W' = (1 + b) W D W^T - b (W D W^T)(W D W^T)^T (W D W^T)``, ``D = clamp(d, 0, 1)``.
    """

    def __init__(self, num_nodes: int, width: int, beta: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_nodes,), 0.8))
        self.weight = nn.Parameter(torch.eye(width))
        self.scale = nn.Parameter(torch.ones(width))
        self.beta = beta

    def channel_weight(self) -> torch.Tensor:
        weight = (self.weight * self.scale.clamp(0.0, 1.0)) @ self.weight.t()
        return (1 + self.beta) * weight - self.beta * weight @ weight.t() @ weight

    def forward(self, h, h0, support, mask, time_gate: TemporalGate) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return (alpha / 2 * graph_message(h, support, mask) - h + time_gate.node(h) - h
                + h @ self.channel_weight() - h + h0)


class EdgeODE(nn.Module):
    """Edge ODE function (Eq. 17, official form): ``sigmoid(a)/2 (A + M) E - E + T_e(E) - E + E(0)``."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_nodes,), 0.8))

    def forward(self, e, e0, support, mask, time_gate: TemporalGate) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return alpha / 2 * graph_message(e, support, mask) - e + time_gate.edge(e) - e + e0


def euler(func, state: torch.Tensor, steps: int, *args) -> list[torch.Tensor]:
    """Unit-step explicit Euler from ``state``; returns the ``steps`` states after ``t = 0``.

    The constant ``H(0)`` term of the derivative is the detached initial state.
    """
    initial = state.detach()
    states = []
    for _ in range(steps):
        state = state + func(state, initial, *args)
        states.append(state)
    return states


class HeadAttention(nn.Module):
    """Multi-head scaled dot-product self-attention over the node axis, without output projection."""

    def __init__(self, dim_in: int, dim_k: int, dim_v: int, heads: int) -> None:
        super().__init__()
        if dim_k % heads or dim_v % heads:
            raise ValueError("attention key and value widths must be multiples of the head count")
        self.heads = heads
        self.query = nn.Linear(dim_in, dim_k)
        self.key = nn.Linear(dim_in, dim_k)
        self.value = nn.Linear(dim_in, dim_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        q = self.query(x).view(batch, nodes, self.heads, -1).transpose(1, 2)
        k = self.key(x).view(batch, nodes, self.heads, -1).transpose(1, 2)
        v = self.value(x).view(batch, nodes, self.heads, -1).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        return (weights @ v).transpose(1, 2).reshape(batch, nodes, -1)


class MultiODEBlock(nn.Module):
    """Multi ODE-GNN block (Fig. 3, Algorithm 1 lines 4-26) on ``[B, N, T, C]``."""

    def __init__(self, num_nodes: int, seq_len: int, width: int, local_chunks: int,
                 local_heads: int, beta: float) -> None:
        super().__init__()
        self.local_chunks = local_chunks
        self.local_steps = seq_len // local_chunks
        self.global_time = TemporalGate(seq_len)  # shared by the global and edge ODEs
        self.local_time = TemporalGate(1)  # shared by the local ODEs
        self.mask = nn.Parameter(torch.randn(num_nodes, num_nodes))  # shared spatial weight M
        self.global_ode = NodeODE(num_nodes, width, beta)
        self.edge_ode = EdgeODE(num_nodes)
        self.local_odes = nn.ModuleList(NodeODE(num_nodes, width, beta) for _ in range(local_chunks))
        self.local_attention = HeadAttention(seq_len * width, 2 * local_chunks * width,
                                             local_chunks * width, local_heads)
        self.local_proj = nn.Linear(width, width)
        self.global_proj = nn.Linear(width, width)
        self.edge_proj = nn.Linear(num_nodes, width)
        self.residual_proj = nn.Linear(width, width)
        self.clip = nn.Parameter(torch.randn(1))  # learnable band e of Eq. (18)
        self.mix = nn.Parameter(torch.randn(2))  # update weights of Eq. (20)

    @staticmethod
    def edge_features(h: torch.Tensor) -> torch.Tensor:
        """``H_e(0)[b, i, j, t] = mean_c H[b, i, t] + mean_c H[b, j, t]``."""
        mean = h.mean(dim=-1)
        return mean.unsqueeze(1) + mean.unsqueeze(2)

    def local_message(self, h: torch.Tensor, support, mask) -> torch.Tensor:
        """Eqs. (11)-(15): attention tokens start ``T / K``-step local ODEs; outputs are concatenated."""
        batch, nodes, steps, width = h.shape
        tokens = self.local_attention(h.reshape(batch, nodes, steps * width))
        tokens = tokens.view(batch, nodes, self.local_chunks, width)
        chunks = []
        for i, ode in enumerate(self.local_odes):
            states = euler(ode, tokens[:, :, i:i + 1], self.local_steps, support, mask, self.local_time)
            chunks.append(torch.cat(states, dim=2))
        return torch.cat(chunks, dim=2)

    @staticmethod
    def message_filter(local: torch.Tensor, global_: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        """Eq. (18) in the official order: cap above ``GM + e``, then raise below ``GM - e``."""
        local = torch.where(global_ + clip - local < 0, global_ + clip, local)
        return torch.where(global_ - clip - local > 0, global_ - clip, local)

    @staticmethod
    def aggregate(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Eq. (19) with the official sigmoid gate: ``1/(2K) sum_m sum_{n != m} p_m * sigmoid(p_n)``."""
        terms = (p0, p1, p2)
        total = sum(terms[m] * torch.sigmoid(terms[n]) for m in range(3) for n in range(3) if n != m)
        return total / 6

    def forward(self, h: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        mask = self.mask
        global_ = euler(self.global_ode, h, 1, support, mask, self.global_time)[-1]
        edge = euler(self.edge_ode, self.edge_features(h), 1, support, mask, self.global_time)[-1]
        local = self.message_filter(self.local_message(h, support, mask), global_, self.clip[0])
        fused = self.aggregate(self.local_proj(local), self.global_proj(global_),
                               self.edge_proj(edge.permute(0, 2, 3, 1)))
        # Eq. (20): gated residual from the block input, learnable weights.
        return F.relu(self.mix[0] * torch.sigmoid(self.residual_proj(h)) + self.mix[1] * fused)


class GRAMODELayer(nn.Module):
    """TCN -> multi ODE-GNN block -> TCN, then BatchNorm with nodes as channels."""

    def __init__(self, in_dim: int, num_nodes: int, seq_len: int, width: int, tcn_mid_dim: int,
                 kernel_size: int, dropout: float, local_chunks: int, local_heads: int,
                 beta: float) -> None:
        super().__init__()
        channels = (width, tcn_mid_dim, width)
        self.tcn_in = TemporalConvNet(in_dim, channels, kernel_size, dropout)
        self.block = MultiODEBlock(num_nodes, seq_len, width, local_chunks, local_heads, beta)
        self.tcn_out = TemporalConvNet(width, channels, kernel_size, dropout)
        self.norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tcn_out(self.block(self.tcn_in(x), support)))


class Model(nn.Module):
    """``x_enc [B, L, N]`` (plus optional covariates) -> ``[B, H, N]`` point forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 1,
        hidden_dim: int = 64,
        tcn_mid_dim: int = 32,
        kernel_size: int = 2,
        dropout: float = 0.2,
        channels: int = 3,
        layers: int = 2,
        local_chunks: int = 4,
        local_heads: int = 32,
        output_heads: int = 12,
        graph_alpha: float = 0.8,
        ode_beta: float = 0.6,
        steps_per_day: int = 288,
        semantic_sigma: float = 0.1,
        semantic_threshold: float = 0.6,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("lengths and node counts must be positive")
        if seq_len % local_chunks:
            raise ValueError("seq_len must be a multiple of local_chunks")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim = input_dim
        self.graph_alpha = graph_alpha
        self.steps_per_day = steps_per_day
        self.semantic_sigma, self.semantic_threshold = semantic_sigma, semantic_threshold
        self.huber_delta = huber_delta  # read by the training objective (Eq. 24)

        adj = np.zeros((num_nodes, num_nodes)) if adj_mx is None else square_adjacency(adj_mx, num_nodes, dtype=np.float64)
        support = torch.as_tensor(normalized_support(adj, graph_alpha), dtype=torch.float32)
        self.register_buffer("spatial_support", support)
        # Filled from the training split by ``fit_semantic_graph``; no edges until then.
        empty = normalized_support(np.zeros((num_nodes, num_nodes)), graph_alpha)
        self.register_buffer("semantic_support", torch.as_tensor(empty, dtype=torch.float32))

        def stream() -> nn.ModuleList:
            return nn.ModuleList(
                nn.ModuleList(
                    GRAMODELayer(input_dim if layer == 0 else hidden_dim, num_nodes, seq_len,
                                 hidden_dim, tcn_mid_dim, kernel_size, dropout, local_chunks,
                                 local_heads, ode_beta)
                    for layer in range(layers)
                )
                for _ in range(channels)
            )

        self.spatial_stream = stream()
        self.semantic_stream = stream()
        streams = 2 * channels
        self.output_attention = HeadAttention(seq_len * hidden_dim * streams, streams * hidden_dim,
                                              pred_len, output_heads)

    @torch.no_grad()
    def fit_semantic_graph(self, series: torch.Tensor) -> torch.Tensor:
        """Build the DTW graph (Eq. 2, official transform) from a ``[T, N]`` training series."""
        if series.ndim != 2 or series.shape[1] != self.num_nodes:
            raise ValueError(f"series must be [T, {self.num_nodes}]")
        profiles = daily_profiles(series.detach().cpu().to(torch.float64), self.steps_per_day)
        adjacency = semantic_adjacency(dtw_distances(profiles), self.semantic_sigma, self.semantic_threshold)
        support = normalized_support(adjacency.numpy(), self.graph_alpha)
        self.semantic_support.copy_(torch.as_tensor(support, dtype=self.semantic_support.dtype))
        return adjacency

    def node_features(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """``[B, L, N]`` values (plus covariates when ``input_dim > 1``) -> ``[B, N, L, F]``."""
        if self.input_dim == 1:
            data = x_enc.unsqueeze(-1)
        else:
            data = to_spatiotemporal(x_enc, x_mark_enc)
            if data.shape[-1] < self.input_dim:
                raise ValueError(f"expected at least {self.input_dim} node features")
            data = data[..., : self.input_dim]
        return data.permute(0, 2, 1, 3)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.num_nodes)
        x = self.node_features(x_enc, x_mark_enc)
        outputs = []
        for stream, support in ((self.spatial_stream, self.spatial_support),
                                (self.semantic_stream, self.semantic_support)):
            for channel in stream:
                h = x
                for layer in channel:
                    h = layer(h, support)
                outputs.append(h)
        stacked = torch.stack(outputs, dim=-1)  # [B, N, T, C, 2 * channels]
        flat = stacked.reshape(stacked.shape[0], self.num_nodes, -1)
        return self.output_attention(flat).transpose(1, 2)


def training_series(dataset, num_nodes: int) -> torch.Tensor | None:
    """The ``[T, N]`` value series of the training split behind ``dataset``, or ``None``.

    Index-windowed datasets keep the full series and the window ends in ``idx``;
    only the history span of the training windows is returned.
    """
    data = getattr(dataset, "data", None)
    if data is None:
        return None
    values = torch.as_tensor(np.asarray(data), dtype=torch.float64)
    if values.ndim == 3:
        values = values[..., 0]
    idx = getattr(dataset, "idx", None)
    if idx is not None and len(idx):
        ends = np.asarray(idx).reshape(-1)
        start = max(int(ends.min()) - int(getattr(dataset, "seq_len", 1)) + 1, 0)
        values = values[start: int(ends.max()) + 1]
    if values.ndim != 2 or values.shape[1] != num_nodes or values.shape[0] < 2:
        return None
    return values


def fit_from_loader(model: Model, train_loader) -> bool:
    """Fit the DTW graph from the loader's training split; warn and keep no edges otherwise."""
    series = training_series(getattr(train_loader, "dataset", None), model.num_nodes)
    if series is None:
        warnings.warn("GRAMODE: no ordered training series available; the DTW graph has no edges",
                      stacklevel=2)
        return False
    model.fit_semantic_graph(series)
    return True

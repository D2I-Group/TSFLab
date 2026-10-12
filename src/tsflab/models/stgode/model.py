"""Clean-room STGODE with spatial/semantic tensor-ODE branches."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


def _row_normalize(value: np.ndarray) -> np.ndarray:
    value = value + np.eye(value.shape[0], dtype=np.float32)
    return value / np.maximum(value.sum(-1, keepdims=True), 1e-6)


class TensorODEBlock(nn.Module):
    """Explicit Euler integration of simultaneous temporal and graph dynamics."""

    def __init__(self, width: int, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.temporal = nn.Conv2d(width, width, (1, 3))
        self.channel = nn.Linear(width, width)
        self.step_size = nn.Parameter(torch.tensor(0.25))

    def forward(self, state: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            channels = state.permute(0, 3, 2, 1)
            temporal = self.temporal(F.pad(channels, (2, 0, 0, 0))).permute(0, 3, 2, 1)
            spatial = torch.einsum("nm,btmd->btnd", graph, state)
            derivative = torch.tanh(self.channel(spatial) + temporal - state)
            state = state + torch.sigmoid(self.step_size) * derivative
        return state


class ODEBranch(nn.Module):
    def __init__(self, width: int, steps: int) -> None:
        super().__init__()
        self.pre_filter = nn.Conv2d(width, width, (1, 3), dilation=(1, 2))
        self.ode = TensorODEBlock(width, steps)
        self.post_filter = nn.Conv2d(width, width, (1, 3), dilation=(1, 4))
        self.norm = nn.LayerNorm(width)

    @staticmethod
    def _causal(conv, x, dilation):
        return conv(F.pad(x, (2 * dilation, 0, 0, 0)))

    def forward(self, x, graph):
        channels = x.permute(0, 3, 2, 1)
        first = torch.tanh(self._causal(self.pre_filter, channels, 2)).permute(0, 3, 2, 1)
        evolved = self.ode(first, graph)
        second = torch.tanh(self._causal(self.post_filter, evolved.permute(0, 3, 2, 1), 4)).permute(
            0, 3, 2, 1
        )
        return self.norm(x + second)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        hidden_dim: int = 32,
        ode_steps: int = 2,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("invalid STGODE dimensions")
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = (
            seq_len,
            pred_len,
            num_nodes,
            input_dim,
        )
        adj = (
            np.eye(num_nodes, dtype=np.float32)
            if adj_mx is None
            else square_adjacency(adj_mx, num_nodes)
        )
        spatial = _row_normalize(adj)
        profiles = spatial @ spatial.T
        semantic = _row_normalize(np.maximum(profiles, 0).astype(np.float32))
        self.register_buffer("spatial_graph", torch.from_numpy(spatial))
        self.register_buffer("semantic_graph", torch.from_numpy(semantic))
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.spatial_branch = ODEBranch(hidden_dim, ode_steps)
        self.semantic_branch = ODEBranch(hidden_dim, ode_steps)
        self.branch_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.forecast = nn.Linear(seq_len * hidden_dim, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.num_nodes)
        st = to_spatiotemporal(x_enc, x_mark_enc)
        if st.shape[-1] < self.input_dim:
            st = torch.cat((st, st.new_zeros(*st.shape[:-1], self.input_dim - st.shape[-1])), -1)
        base = self.input_projection(st[..., : self.input_dim])
        spatial = self.spatial_branch(base, self.spatial_graph)
        semantic = self.semantic_branch(base, self.semantic_graph)
        gate = torch.sigmoid(self.branch_gate(torch.cat((spatial, semantic), -1)))
        fused = gate * spatial + (1 - gate) * semantic
        return self.forecast(fused.transpose(1, 2).flatten(2)).transpose(1, 2)

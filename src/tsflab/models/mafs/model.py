"""Clean-room Multi-Agent Forecasting System (MAFS)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _topology_mask(num_agents: int, topology: str) -> torch.Tensor:
    mask = torch.zeros(num_agents, num_agents)
    if topology == "fully-connected":
        mask.fill_(1)
    elif topology == "ring":
        for index in range(num_agents):
            mask[index, (index - 1) % num_agents] = 1
            mask[index, (index + 1) % num_agents] = 1
    elif topology == "chain":
        for index in range(num_agents - 1):
            mask[index, index + 1] = mask[index + 1, index] = 1
    else:  # star
        mask[0, :] = 1
        mask[:, 0] = 1
    return mask


class AgentEncoderLayer(nn.Module):
    """An iTransformer-style encoder over variate tokens."""

    def __init__(self, dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + self.dropout(attended))
        return self.feed_forward_norm(x + self.dropout(self.feed_forward(x)))


class Model(nn.Module):
    """Specialized variate-token agents with graph communication and AVA voting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        num_agents: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        topology: str = "star",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_agents = num_agents
        self.input_embeddings = nn.ModuleList(
            [nn.Linear(seq_len, d_model) for _ in range(num_agents)]
        )
        self.agent_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [AgentEncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
                )
                for _ in range(num_agents)
            ]
        )
        self.communication = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)]
        )
        self.edge_logits = nn.Parameter(torch.zeros(num_agents, num_agents))
        self.register_buffer("topology_mask", _topology_mask(num_agents, topology))
        self.confidence_gate = nn.Linear(num_agents * d_model, num_agents * d_model)
        self.voter = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, num_agents)
        )
        self.agent_heads = nn.ModuleList(
            [nn.Linear(d_model, pred_len) for _ in range(num_agents)]
        )
        self.final_head = nn.Linear(d_model, pred_len)
        self.last_adjacency: torch.Tensor | None = None
        self.last_voting_weights: torch.Tensor | None = None

    def normalized_adjacency(self, adaptive: bool = True) -> torch.Tensor:
        """Equation (5): masked, symmetrized, self-looped graph normalization."""
        weighted = (
            torch.sigmoid(self.edge_logits) * self.topology_mask
            if adaptive
            else self.topology_mask
        )
        symmetric = (weighted + weighted.T) / 2
        adjacency = symmetric + torch.eye(
            self.num_agents, dtype=weighted.dtype, device=weighted.device
        )
        degree = adjacency.sum(-1).clamp_min(1e-6).rsqrt()
        return degree[:, None] * adjacency * degree[None, :]

    def specialization_targets(self, target: torch.Tensor) -> list[torch.Tensor]:
        """Homogeneous multi-scale targets from Appendix A."""
        if target.ndim != 3 or target.shape[1:] != (self.pred_len, self.enc_in):
            raise ValueError("target has the wrong forecasting shape")
        return [
            target[:, : max(1, self.pred_len * (index + 1) // self.num_agents)]
            for index in range(self.num_agents)
        ]

    def agent_representations(
        self, x: torch.Tensor, adaptive_topology: bool = True
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )
        variates = x.transpose(1, 2)
        states = torch.stack(
            [embedding(variates) for embedding in self.input_embeddings], dim=1
        )
        adjacency = self.normalized_adjacency(adaptive_topology)
        for layer_index, communication in enumerate(self.communication):
            encoded = torch.stack(
                [
                    self.agent_layers[agent_index][layer_index](states[:, agent_index])
                    for agent_index in range(self.num_agents)
                ],
                dim=1,
            )
            messages = torch.einsum("ij,bjcd->bicd", adjacency, encoded)
            states = F.gelu(communication(messages))
        self.last_adjacency = adjacency
        return states

    def specialization_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Stage-1 homogeneous prefix objectives with the fixed communication graph."""
        targets = self.specialization_targets(target)
        states = self.agent_representations(x, adaptive_topology=False)
        losses = []
        for index, (head, specialized_target) in enumerate(
            zip(self.agent_heads, targets, strict=True)
        ):
            prediction = head(states[:, index]).transpose(1, 2)
            losses.append(F.mse_loss(prediction[:, : specialized_target.shape[1]], specialized_target))
        return torch.stack(losses).mean()

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        states = self.agent_representations(x_enc, adaptive_topology=True)
        batch, _, channels, dimension = states.shape
        context = torch.einsum("ij,bjcd->bicd", self.normalized_adjacency(), states)
        concatenated = context.permute(0, 2, 1, 3).reshape(batch, channels, -1)
        gates = torch.sigmoid(self.confidence_gate(concatenated)).reshape(
            batch, channels, self.num_agents, dimension
        ).permute(0, 2, 1, 3)
        adjusted = gates * states + (1 - gates) * context

        encoded_input = states.mean(dim=(1, 2))
        voting = torch.softmax(self.voter(encoded_input), -1)
        aggregate = (adjusted * voting[:, :, None, None]).sum(1)
        representation_forecast = self.final_head(aggregate).transpose(1, 2)

        agent_forecasts = torch.stack(
            [
                head(adjusted[:, index]).transpose(1, 2)
                for index, head in enumerate(self.agent_heads)
            ],
            dim=1,
        )
        voted_forecast = (agent_forecasts * voting[:, :, None, None]).sum(1)
        self.last_voting_weights = voting
        return representation_forecast + voted_forecast

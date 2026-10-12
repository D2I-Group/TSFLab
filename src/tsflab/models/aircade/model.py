"""Clean-room AirCade from the 2025 causal-decoupling paper.

This file maps paper Eqs. (1)--(13) directly to local modules: domain-knowledge
prompts, four-path DK-MSA, historical causal decoupling (Cade), future causal
diffusion (Cadi), and learnable multi-environment intervention masks.  Temporal
and spatial stages remain distinct so their axes and interventions cannot be
silently confused.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from tsflab.models._components.marks import (
    coerce_time_length,
    future_time_features,
    to_spatiotemporal,
)
from tsflab.models._inputs import check_window


class DomainKnowledgeAttention(nn.Module):
    """Paper Eqs. (2)--(7): direct, inverse, and adaptive paths."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        axis_length: int,
        adaptive_dim: int,
        environments: int,
        gated: bool,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.adaptive_left = nn.Parameter(torch.randn(axis_length, adaptive_dim) * 0.02)
        self.adaptive_right = nn.Parameter(torch.randn(axis_length, adaptive_dim) * 0.02)
        self.intervention_logits = nn.Parameter(
            torch.zeros(environments, axis_length, axis_length)
        )
        self.output = nn.Linear(d_model * 4, d_model)
        self.gated = gated
        if gated:
            self.signal = nn.Linear(d_model, d_model)
            self.gate = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch, length, width = query.shape
        if key.shape[:2] != (batch, length) or value.shape[:2] != (batch, length):
            raise ValueError("DK-MSA query, key, and value axes must match")
        reshape = lambda item: item.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = map(reshape, (self.query(query), self.key(key), self.value(value)))
        direct = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        inverse = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.head_dim)
        adaptive = torch.relu(self.adaptive_left @ self.adaptive_right.transpose(0, 1))
        reverse_adaptive = torch.relu(self.adaptive_right @ self.adaptive_left.transpose(0, 1))
        intervention = torch.sigmoid(self.intervention_logits).mean(0)
        matrices = (
            direct.softmax(-1),
            adaptive.softmax(-1).view(1, 1, length, length),
            inverse.softmax(-1).transpose(-1, -2),
            reverse_adaptive.softmax(-1).transpose(-1, -2).view(1, 1, length, length),
        )
        paths = [torch.matmul(matrix * intervention, v) for matrix in matrices]
        combined = torch.cat(paths, dim=-1).transpose(1, 2).reshape(batch, length, width * 4)
        combined = self.output(combined)
        if self.gated:
            combined = torch.tanh(self.signal(combined)) * torch.sigmoid(self.gate(combined))
        return combined


class CausalLayer(nn.Module):
    """Cade/Cadi residual attention and MLP, paper Eqs. (8)--(11)."""

    def __init__(self, attention: DomainKnowledgeAttention, d_model: int) -> None:
        super().__init__()
        self.attention = attention
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, meteorology: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        value = self.attention_norm(
            value + self.attention(meteorology, meteorology, value)
        )
        return self.output_norm(value + self.feed_forward(value))


class Model(nn.Module):
    """Spatiotemporal causal-decoupling air-quality forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int = 2,
        d_model: int = 32,
        prompt_dim: int = 8,
        adaptive_dim: int = 8,
        num_heads: int = 4,
        temporal_layers: int = 2,
        spatial_layers: int = 2,
        environments: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cov_dim = cov_dim

        self.value_embedding = nn.Linear(1, d_model - prompt_dim * 2)
        self.past_weather_embedding = nn.Linear(cov_dim, d_model - prompt_dim * 2)
        self.future_weather_embedding = nn.Linear(cov_dim, d_model - prompt_dim * 2)
        self.past_time_prompt = nn.Parameter(torch.randn(seq_len, prompt_dim) * 0.02)
        self.future_time_prompt = nn.Parameter(torch.randn(pred_len, prompt_dim) * 0.02)
        self.station_prompt = nn.Parameter(torch.randn(enc_in, prompt_dim) * 0.02)

        def attention(length: int, gated: bool) -> DomainKnowledgeAttention:
            return DomainKnowledgeAttention(
                d_model, num_heads, length, adaptive_dim, environments, gated
            )

        self.temporal_cade = nn.ModuleList(
            CausalLayer(attention(seq_len, True), d_model) for _ in range(temporal_layers)
        )
        self.spatial_cade = nn.ModuleList(
            CausalLayer(attention(enc_in, False), d_model) for _ in range(spatial_layers)
        )
        self.history_to_future = nn.Linear(seq_len, pred_len)
        self.temporal_cadi = nn.ModuleList(
            CausalLayer(attention(pred_len, True), d_model) for _ in range(temporal_layers)
        )
        self.spatial_cadi = nn.ModuleList(
            CausalLayer(attention(enc_in, False), d_model) for _ in range(spatial_layers)
        )
        self.predictor = nn.Linear(d_model, 1)

    def _prompted(
        self, embedding: torch.Tensor, time_prompt: torch.Tensor
    ) -> torch.Tensor:
        batch, steps, nodes, _ = embedding.shape
        time = time_prompt.view(1, steps, 1, -1).expand(batch, steps, nodes, -1)
        station = self.station_prompt.view(1, 1, nodes, -1).expand(batch, steps, nodes, -1)
        return torch.cat([embedding, time, station], dim=-1)

    @staticmethod
    def _temporal(layer: CausalLayer, weather: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = value.shape
        flatten = lambda item: item.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        result = layer(flatten(weather), flatten(value))
        return result.reshape(batch, nodes, steps, width).permute(0, 2, 1, 3)

    @staticmethod
    def _spatial(layer: CausalLayer, weather: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = value.shape
        result = layer(
            weather.reshape(batch * steps, nodes, width),
            value.reshape(batch * steps, nodes, width),
        )
        return result.reshape(batch, steps, nodes, width)

    def _causal_stack(
        self, temporal: nn.ModuleList, spatial: nn.ModuleList, weather: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        for layer in temporal:
            value = self._temporal(layer, weather, value)
        for layer in spatial:
            value = self._spatial(layer, weather, value)
        return value

    def _history(self, x_enc, x_mark_enc):
        check_window(x_enc, self.seq_len, self.enc_in)
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], self.seq_len, 6))
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] != 1 + self.cov_dim:
            raise ValueError(f"AirCade expects exactly {self.cov_dim} historical covariates")
        return history, x_mark_enc

    def _covariates(self, x_enc, x_mark_enc, x_mark_dec):
        history, x_mark_enc = self._history(x_enc, x_mark_enc)
        future_marks = x_mark_enc if x_mark_dec is None else x_mark_dec
        future_marks = coerce_time_length(future_marks, self.pred_len)
        future = future_time_features(future_marks, self.enc_in)
        if future.shape[-1] != self.cov_dim:
            raise ValueError(f"AirCade expects exactly {self.cov_dim} future covariates")
        return history, future

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        history, future = self._covariates(x_enc, x_mark_enc, x_mark_dec)

        value = self._prompted(self.value_embedding(history[..., :1]), self.past_time_prompt)
        past_weather = self._prompted(
            self.past_weather_embedding(history[..., 1:]), self.past_time_prompt
        )
        value = self._causal_stack(self.temporal_cade, self.spatial_cade, past_weather, value)

        value = self.history_to_future(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        future_weather = self._prompted(
            self.future_weather_embedding(future), self.future_time_prompt
        )
        value = self._causal_stack(self.temporal_cadi, self.spatial_cadi, future_weather, value)
        return self.predictor(value).squeeze(-1)

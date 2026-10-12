"""Independent TiRex-style decoder-only probabilistic forecaster.

The implementation follows the paper's z-score scaling, value/missing-mask
patch tokens, recurrent scalar-memory blocks, missing future-token decoding,
and multi-patch quantile forecast. The optimized xLSTM kernels and published
pre-trained weights are not reproduced.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.quantile_head import QuantileHead
from tsflab.models._inputs import check_window


_DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class ResidualProjection(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.skip = nn.Linear(input_width, output_width)
        self.residual = nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.SiLU(),
            nn.Linear(output_width, output_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.residual(x)


class ScalarMemory(nn.Module):
    """Stabilized scalar LSTM memory with exponential input/forget gates."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gates = nn.Linear(width, 4 * width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, _, width = sequence.shape
        cell = sequence.new_zeros(batch, width)
        normalizer = sequence.new_zeros(batch, width)
        stabilizer = sequence.new_full((batch, width), -torch.inf)
        outputs: list[torch.Tensor] = []
        for item in sequence.unbind(dim=1):
            proposal, log_input, log_forget, output_gate = self.gates(item).chunk(4, dim=-1)
            next_stabilizer = torch.maximum(log_forget + stabilizer, log_input)
            forget = torch.exp(log_forget + stabilizer - next_stabilizer)
            input_gate = torch.exp(log_input - next_stabilizer)
            cell = forget * cell + input_gate * torch.tanh(proposal)
            normalizer = forget * normalizer + input_gate
            hidden = torch.sigmoid(output_gate) * cell / normalizer.clamp_min(1e-6)
            outputs.append(hidden)
            stabilizer = next_stabilizer
        return torch.stack(outputs, dim=1)


class ScalarLSTMBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.recurrent_norm = nn.RMSNorm(width)
        self.recurrent = ScalarMemory(width)
        self.feedforward_norm = nn.RMSNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.recurrent(self.recurrent_norm(tokens))
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        d_model: int = 64,
        patch_len: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.output_type = "quantile"
        self.patch_len = patch_len
        self.future_patches = math.ceil(pred_len / patch_len)
        levels = list(quantile_levels) if quantile_levels else _DEFAULT_LEVELS
        self.input_projection = ResidualProjection(2 * patch_len, d_model)
        self.blocks = nn.ModuleList(
            ScalarLSTMBlock(d_model, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.output_projection = ResidualProjection(d_model, patch_len * d_model)
        self.quantile_head = QuantileHead(levels, in_features=d_model)

    @staticmethod
    def contiguous_patch_mask(
        batch_size: int,
        patch_count: int,
        max_span: int,
        probability: float,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample blockwise CPM masks following the paper's repeat construction."""
        if max_span < 1 or not 0 <= probability <= 1:
            raise ValueError("max_span must be positive and probability must be in [0, 1]")
        blocks = math.ceil(patch_count / max_span)
        sampled = torch.rand(batch_size, blocks, device=device) < probability
        return sampled.repeat_interleave(max_span, dim=1)[:, :patch_count]

    def _history_tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        values = normalized.transpose(1, 2).reshape(-1, self.seq_len)
        padded = math.ceil(self.seq_len / self.patch_len) * self.patch_len
        if padded > self.seq_len:
            values = F.pad(values, (padded - self.seq_len, 0))
        patches = values.unfold(-1, self.patch_len, self.patch_len)
        observed = torch.ones_like(patches)
        return self.input_projection(torch.cat([patches, observed], dim=-1))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized, mean, scale = standardize(x_enc, 1e-5, detach_mean=True, detach_std=True)
        history = self._history_tokens(normalized)
        missing = normalized.new_zeros(
            history.shape[0], self.future_patches, 2 * self.patch_len
        )
        tokens = torch.cat([history, self.input_projection(missing)], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        future = self.final_norm(tokens[:, -self.future_patches :])
        features = self.output_projection(future).reshape(
            x_enc.shape[0], self.enc_in, self.future_patches * self.patch_len, -1
        )[:, :, : self.pred_len]
        quantiles = self.quantile_head(features.permute(0, 2, 1, 3))
        quantiles = destandardize(quantiles, mean.unsqueeze(-1), scale.unsqueeze(-1))
        if self.features == "MS":
            quantiles = quantiles[:, :, -1:]
        return quantiles


__all__ = ["Model", "ResidualProjection", "ScalarLSTMBlock", "ScalarMemory"]

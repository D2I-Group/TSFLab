"""Local AutoTimes implementation (GPT-2 variant) from the paper and pinned official code.

AutoTimes cuts each channel's history into non-overlapping segment tokens,
embeds every segment into the LLM's token space, runs a frozen decoder-only
LLM, and projects every output position back to the *next* segment. Training
supervises all next-segment predictions at once; inference rolls the
next-segment forecast forward autoregressively.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.gpt2_backbone import GPT2Backbone, GPT2Config
from tsflab.models._components.revin import RevIN
from tsflab.models._components.segment_mlp import SegmentMLP


class Model(nn.Module):
    """AutoTimes with a frozen GPT-2 as the autoregressive token-transition model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        token_len: int = 96,
        mlp_hidden_dim: int = 512,
        mlp_hidden_layers: int = 2,
        mlp_activation: str = "tanh",
        dropout: float = 0.1,
        llm_layers: int = 12,
    ) -> None:
        super().__init__()
        if seq_len % token_len:
            raise ValueError("seq_len must be a multiple of token_len (context = N segments)")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.token_len = token_len
        self.token_num = seq_len // token_len
        self.llm = GPT2Backbone(GPT2Config(n_layer=llm_layers))
        width = self.llm.config.n_embd
        if self.token_num > self.llm.config.n_positions:
            raise ValueError("seq_len / token_len exceeds the LLM context")
        # The LLM is entirely frozen; only the segment embedding/projection train.
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        self.normalizer = RevIN(1, affine=False)
        self.encoder = SegmentMLP(token_len, width, mlp_hidden_dim, mlp_hidden_layers,
                                  dropout, mlp_activation)
        self.decoder = SegmentMLP(width, token_len, mlp_hidden_dim, mlp_hidden_layers,
                                  dropout, mlp_activation)

    def next_segments(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Map segments ``s_1..s_N`` of ``[B, N*S, C]`` to predictions ``s_2..s_{N+1}``.

        Output position block ``i`` holds the forecast of the segment that
        follows input segment ``i`` (Eq. 6-7), in the input's scale.
        """
        if x_enc.ndim != 3 or x_enc.shape[1] % self.token_len:
            raise ValueError(f"AutoTimes expects [batch, N * {self.token_len}, channels]")
        batch, length, channels = x_enc.shape
        # Channel independence: every channel is one univariate sequence.
        series = x_enc.permute(0, 2, 1).reshape(batch * channels, length, 1)
        series = self.normalizer(series, "norm")
        # Eq. 2-3: non-overlapping segments s_i -> SE_i.
        segments = series.squeeze(-1).unfold(-1, self.token_len, self.token_len)
        hidden = self.llm(self.encoder(segments))  # Eq. 6: frozen LLM layers
        predicted = self.decoder(hidden).reshape(batch * channels, length, 1)  # Eq. 7
        predicted = self.normalizer(predicted, "denorm")
        return predicted.reshape(batch, channels, length).permute(0, 2, 1)

    def next_segment_loss(self, x_enc: torch.Tensor, future: torch.Tensor, criterion) -> torch.Tensor:
        """Token-wise next-segment objective (Eq. 8) against the shifted ground truth.

        The target is the input shifted by one segment, completed with the first
        future values; if fewer than ``token_len`` future values are available
        only the covered positions are supervised.
        """
        available = min(self.token_len, future.shape[1])
        target = torch.cat([x_enc, future[:, :available]], dim=1)[:, self.token_len:]
        predicted = self.next_segments(x_enc)[:, : target.shape[1]]
        return criterion(predicted, target)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        """Autoregressive rolling forecast (Eq. 9) of ``pred_len`` steps."""
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"AutoTimes expects [batch, {self.seq_len}, channels]")
        window, outputs = x_enc, []
        for _ in range(math.ceil(self.pred_len / self.token_len)):
            segment = self.next_segments(window)[:, -self.token_len:]
            outputs.append(segment)
            # Slide the context: drop the oldest segment, append the generated one.
            window = torch.cat([window[:, self.token_len:], segment], dim=1)
        return torch.cat(outputs, dim=1)[:, : self.pred_len]

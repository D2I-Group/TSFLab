"""Clean-room CMoS implementation from the ICML paper."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


def periodic_correlation_initialization(
    input_chunks: int, output_chunks: int, period_chunks: int
) -> torch.Tensor:
    """Construct the periodic peaks described by Section 3.3/Algorithm 1."""
    if period_chunks < 1:
        raise ValueError("period_chunks must be positive")
    weights = torch.zeros(input_chunks, output_chunks)
    for source in range(input_chunks):
        for target in range(output_chunks):
            if (input_chunks + target - source) % period_chunks == 0:
                weights[source, target] = 1.0 / period_chunks
    return weights


class CorrelationMixer(nn.Module):
    """Channel-conditioned mixture of K chunk-correlation matrices (Eqs. 3--5)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        chunk_size: int,
        num_correlations: int,
        kernel_size: int,
        period: int | None,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.channels = channels
        input_chunks = seq_len // chunk_size
        output_chunks = pred_len // chunk_size
        self.correlations = nn.Parameter(
            torch.empty(num_correlations, input_chunks, output_chunks)
        )
        nn.init.xavier_uniform_(self.correlations)
        if period is not None:
            period_chunks = period // chunk_size
            with torch.no_grad():
                self.correlations[0].copy_(
                    periodic_correlation_initialization(
                        input_chunks, output_chunks, period_chunks
                    )
                )
        stride = max(1, kernel_size // 2)
        summary_length = (seq_len - kernel_size) // stride + 1
        self.aggregators = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size, stride=stride) for _ in range(channels)]
        )
        self.allocator = nn.Linear(summary_length, num_correlations)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, channels = values.shape
        channel_first = values.transpose(1, 2)
        summaries = torch.stack(
            [
                aggregator(channel_first[:, index : index + 1]).squeeze(1)
                for index, aggregator in enumerate(self.aggregators)
            ],
            dim=1,
        )
        mixture = torch.softmax(self.allocator(summaries), dim=-1)
        chunks = channel_first.reshape(batch, channels, -1, self.chunk_size)
        candidates = torch.einsum("bcis,kio->bckos", chunks, self.correlations)
        forecast = torch.einsum("bckos,bck->bcos", candidates, mixture)
        return forecast.reshape(batch, channels, -1).transpose(1, 2)


class Model(nn.Module):
    """CMoS with correlation mixing and optional periodicity injection."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        seg_size: int = 4,
        num_map: int = 3,
        kernel_size: int = 4,
        period: int | None = None,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("CMoS dimensions must be positive")
        if seq_len % seg_size or pred_len % seg_size:
            raise ValueError("seg_size must divide both seq_len and pred_len")
        if kernel_size > seq_len:
            raise ValueError("kernel_size cannot exceed seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = c_in
        self.revin = RevIN(c_in)
        self.mixer = CorrelationMixer(
            seq_len, pred_len, c_in, seg_size, num_map, kernel_size, period
        )

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
        return self.revin(self.mixer(normalized), "denorm")

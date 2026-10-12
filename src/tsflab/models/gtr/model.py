"""Clean-room Global Temporal Retriever (GTR) implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.marks import cycle_phase
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class GlobalTemporalRetriever(nn.Module):
    """Equations (1)--(5): cycle alignment, 2D fusion, and residual output."""

    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        cycle_length: int,
        local_period: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if cycle_length < seq_len:
            raise ValueError("cycle_length must cover at least one complete input window")
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.cycle_length = cycle_length
        self.global_embedding = nn.Parameter(torch.zeros(cycle_length, enc_in))
        self.reference_projection = nn.Linear(seq_len, seq_len)
        width = 1 + 2 * (local_period // 2)
        self.fusion = nn.Conv2d(1, 1, kernel_size=(2, width), padding=(0, width // 2))
        self.dropout = nn.Dropout(dropout)

    def cycle_indices(
        self, start_index: int | torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        start = torch.as_tensor(start_index, device=device, dtype=torch.long)
        if start.ndim == 0:
            start = start.expand(batch_size)
        if start.shape != (batch_size,):
            raise ValueError("start_index must be a scalar or one value per batch item")
        offsets = torch.arange(self.seq_len, device=device)
        return (start[:, None] + offsets[None, :]).remainder(self.cycle_length)

    def retrieve(self, start_index: int | torch.Tensor, batch_size: int) -> torch.Tensor:
        indices = self.cycle_indices(start_index, batch_size, self.global_embedding.device)
        reference = self.global_embedding[indices]
        return self.reference_projection(reference.transpose(1, 2)).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, start_index: int | torch.Tensor = 0
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        if (length, channels) != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )
        reference = self.retrieve(start_index, batch)
        paired = torch.stack((x, reference), dim=2)
        paired = paired.permute(0, 3, 2, 1).reshape(batch * channels, 1, 2, length)
        pattern = self.fusion(paired).reshape(batch, channels, length).transpose(1, 2)
        return x + self.dropout(pattern)


class Model(nn.Module):
    """GTR followed by the paper's residual two-layer MLP backbone."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        cycle_length: int = 168,
        local_period: int = 24,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, enabled=use_revin)
        self.retriever = GlobalTemporalRetriever(
            seq_len, enc_in, cycle_length, local_period, dropout
        )
        self.input_projection = nn.Linear(seq_len, d_model)
        self.mlp_first = nn.Linear(d_model, d_model)
        self.mlp_second = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, pred_len)

    def forecast_at(
        self,
        x_enc: torch.Tensor,
        start_index: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        enhanced = self.retriever(normalized, start_index)
        latent = self.input_projection(enhanced.transpose(1, 2))
        hidden = torch.nn.functional.gelu(self.mlp_first(latent))
        hidden = torch.nn.functional.gelu(self.mlp_second(hidden)) + latent
        forecast = self.output_projection(self.dropout(hidden)).transpose(1, 2)
        return self.revin(forecast, "denorm")

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        # The retrieved cycle is aligned to the calendar position of the window's first step.
        start = cycle_phase(x_mark_enc, self.retriever.cycle_length, step=0, batch=x_enc.shape[0])
        return self.forecast_at(x_enc, start_index=start.to(x_enc.device))

"""Clean-room local implementation of the Temporal Query Network (TQNet).

TQNet extends CycleNet's per-variable learnable recurrent cycle into a
cross-variable *temporal query*: the phase-aligned cycle window (one vector
per lookback step, drawn from a table indexed by calendar phase) is used as
the *query* of a single attention layer whose keys and values are the raw
(instance-normalized) lookback window. The attended "channel information"
fuses the global periodic prior with local, per-sample observations before a
lightweight two-layer MLP produces the forecast (paper Section 3, Figure 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.periodic_query_bank import PeriodicQueryBank
from tsflab.models._components.revin import RevIN


class Model(nn.Module):
    """Attention-fused temporal-query forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle: int = 24,
        d_model: int = 64,
        dropout: float = 0.1,
        attn_dropout: float = 0.5,
        channel_aggre_heads: int = 4,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths must be positive")
        if seq_len % channel_aggre_heads:
            raise ValueError("seq_len must be divisible by channel_aggre_heads")
        self.seq_len, self.pred_len, self.enc_in, self.cycle = seq_len, pred_len, enc_in, cycle

        self.normalization = RevIN(enc_in, affine=False, enabled=use_revin)
        self.temporal_query_bank = PeriodicQueryBank(cycle, enc_in)
        self.channel_aggregator = nn.MultiheadAttention(
            embed_dim=seq_len, num_heads=channel_aggre_heads, batch_first=True, dropout=attn_dropout
        )
        self.input_proj = nn.Linear(seq_len, d_model)
        self.backbone = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.output_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def _start_phase(self, marks: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        """Calendar phase of the first forecast step (paper's ``cycle_index``)."""
        if marks is None or marks.ndim != 3 or marks.shape[-1] < 6:
            end_phase = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            stamp = marks[:, -1]
            weekday, hour = stamp[:, 3], stamp[:, 4]
            if self.cycle == 7:
                phase = weekday
            elif self.cycle == 168:
                phase = weekday * 24 + hour
            else:
                phase = hour
            end_phase = phase.long().remainder(self.cycle)
        return (end_phase + 1).remainder(self.cycle)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        normalized = self.normalization(x_enc, "norm")
        # (batch, seq_len, channels) -> (batch, channels, seq_len): channels become tokens.
        x_input = normalized.transpose(1, 2)

        start_phase = self._start_phase(x_mark_enc, x_enc.shape[0], x_enc.device)
        query = self.temporal_query_bank(start_phase, self.seq_len).transpose(1, 2)
        channel_information, _ = self.channel_aggregator(query, x_input, x_input, need_weights=False)

        projected = self.input_proj(x_input + channel_information)
        hidden = self.backbone(projected)
        forecast = self.output_proj(hidden + projected).transpose(1, 2)
        return self.normalization(forecast, "denorm")


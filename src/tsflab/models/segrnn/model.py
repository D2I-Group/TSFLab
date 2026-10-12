"""Paper-driven local implementation of segment recurrent forecasting."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from tsflab.models._components.last_value_center import center_on_last_value, restore_last_value


class Model(nn.Module):
    """Encode history segments recurrently and decode future segments in parallel."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 dropout: float = 0.1, seg_len: int = 24) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len, self.pred_len, self.enc_in, self.seg_len = seq_len, pred_len, enc_in, seg_len
        self.history_segments = math.ceil(seq_len / seg_len)
        self.future_segments = math.ceil(pred_len / seg_len)
        self.segment_projection = nn.Sequential(nn.Linear(seg_len, d_model), nn.ReLU())
        self.recurrent = nn.GRU(d_model, d_model, batch_first=True)
        self.relative_position = nn.Parameter(torch.randn(self.future_segments, d_model // 2) * 0.02)
        self.channel_position = nn.Parameter(torch.randn(enc_in, d_model // 2) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.segment_decoder = nn.Linear(d_model, seg_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        centered, level = center_on_last_value(x_enc)
        batch = centered.shape[0]
        padding = self.history_segments * self.seg_len - self.seq_len
        history = centered.transpose(1, 2)
        if padding:
            history = F.pad(history, (padding, 0), mode="replicate")
        segments = history.reshape(batch * self.enc_in, self.history_segments, self.seg_len)
        _, hidden = self.recurrent(self.segment_projection(segments))
        relative = self.relative_position[None, None].expand(batch, self.enc_in, -1, -1)
        channel = self.channel_position[None, :, None].expand(batch, -1, self.future_segments, -1)
        decoder_input = torch.cat((relative, channel), dim=-1)
        decoder_input = decoder_input.reshape(batch * self.enc_in * self.future_segments, 1, -1)
        initial = hidden[-1].reshape(batch, self.enc_in, 1, -1)
        initial = initial.expand(-1, -1, self.future_segments, -1).reshape(1, -1, hidden.shape[-1])
        decoded, _ = self.recurrent(decoder_input, initial.contiguous())
        future = self.segment_decoder(self.dropout(decoded[:, 0]))
        future = future.reshape(batch, self.enc_in, self.future_segments * self.seg_len)
        return restore_last_value(future[..., : self.pred_len].transpose(1, 2), level)

"""Clean-room APN implementation of time-aware patch aggregation.

It implements the paper's soft-window equations, query aggregation, and
query-time decoder. Dense inputs use regular timestamps; callers may pass an
observation-time tensor as the first extra argument. The paper's asynchronous
ragged-data loader and missingness protocol are outside this repository API.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


class _TimeEmbedding(nn.Module):
    def __init__(self, d_time: int) -> None:
        super().__init__()
        self.scale = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_time - 1) if d_time > 1 else None

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        linear = self.scale(times.unsqueeze(-1))
        return linear if self.periodic is None else torch.cat((linear, torch.sin(self.periodic(times.unsqueeze(-1)))), dim=-1)


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 d_time: int = 8, num_patches: int = 8) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.num_patches = num_patches
        self.time_embedding = _TimeEmbedding(d_time)
        self.window_offset = nn.Parameter(torch.zeros(enc_in, num_patches))
        self.log_width = nn.Parameter(torch.full((enc_in, num_patches), math.log(1.0 / num_patches)))
        self.temperature_raw = nn.Parameter(torch.zeros(enc_in))
        self.patch_projection = nn.Linear(1 + d_time, d_model)
        self.patch_position = nn.Parameter(torch.empty(num_patches, d_model))
        self.channel_query = nn.Parameter(torch.empty(enc_in, d_model))
        self.context_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model + d_time, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.normal_(self.patch_position, std=0.02)
        nn.init.normal_(self.channel_query, std=0.02)

    def _times(self, x: torch.Tensor, supplied: torch.Tensor | None) -> torch.Tensor:
        if supplied is None:
            return torch.linspace(0, 1, self.seq_len, device=x.device, dtype=x.dtype).expand(x.size(0), -1)
        if supplied.ndim == 3:
            supplied = supplied[..., 0]
        if supplied.shape != x.shape[:2]:
            raise ValueError("observation times must have shape [batch, sequence] or [batch, sequence, features]")
        lo, hi = supplied.amin(1, keepdim=True), supplied.amax(1, keepdim=True)
        return (supplied - lo) / (hi - lo).clamp_min(torch.finfo(x.dtype).eps)

    def patch_weights(self, times: torch.Tensor) -> torch.Tensor:
        centers = (torch.arange(self.num_patches, device=times.device, dtype=times.dtype) + 0.5) / self.num_patches
        left = centers[None, :] - 0.5 / self.num_patches + self.window_offset
        right = left + self.log_width.exp()
        temp = F.softplus(self.temperature_raw).clamp_min(1e-4)
        t = times[:, None, :, None]
        return torch.sigmoid((right[None, :, None, :] - t) / temp[None, :, None, None]) * torch.sigmoid((t - left[None, :, None, :]) / temp[None, :, None, None])

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        times = self._times(x_enc, x_mark_enc)
        time_features = self.time_embedding(times)
        augmented = torch.cat((x_enc.unsqueeze(-1), time_features[:, :, None, :].expand(-1, -1, self.enc_in, -1)), dim=-1)
        weights = self.patch_weights(times)
        patches = torch.einsum("bclp,blcf->bcpf", weights, augmented)
        patches = patches / weights.sum(2).clamp_min(1e-6).unsqueeze(-1)
        patches = self.patch_projection(patches) + self.patch_position[None, None, :, :]
        scores = torch.einsum("bcpd,cd->bcp", patches, self.channel_query) / math.sqrt(patches.size(-1))
        context = self.context_norm(torch.einsum("bcp,bcpd->bcd", scores.softmax(-1), patches))
        future_times = torch.linspace(1, 2, self.pred_len, device=x_enc.device, dtype=x_enc.dtype).expand(x_enc.size(0), -1)
        future_features = self.time_embedding(future_times)
        decoder_input = torch.cat((context[:, :, None, :].expand(-1, -1, self.pred_len, -1), future_features[:, None, :, :].expand(-1, self.enc_in, -1, -1)), dim=-1)
        return self.decoder(decoder_input).squeeze(-1).transpose(1, 2)

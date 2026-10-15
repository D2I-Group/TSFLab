"""Clean-room Wavelet Patch Mixer from the AAAI 2025 paper."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class OrthogonalWaveletAnalysis(nn.Module):
    """Fixed mathematical analysis filters; returns approximation and details."""

    _FILTERS = {
        "haar": ([2**-0.5, 2**-0.5], [-(2**-0.5), 2**-0.5]),
        "db1": ([2**-0.5, 2**-0.5], [-(2**-0.5), 2**-0.5]),
        "db2": (
            [
                (1 + 3**0.5) / (4 * 2**0.5),
                (3 + 3**0.5) / (4 * 2**0.5),
                (3 - 3**0.5) / (4 * 2**0.5),
                (1 - 3**0.5) / (4 * 2**0.5),
            ],
            [
                (1 - 3**0.5) / (4 * 2**0.5),
                -(3 - 3**0.5) / (4 * 2**0.5),
                (3 + 3**0.5) / (4 * 2**0.5),
                -(1 + 3**0.5) / (4 * 2**0.5),
            ],
        ),
    }

    def __init__(self, wavelet, level):
        super().__init__()
        low, high = self._FILTERS[wavelet]
        self.register_buffer("low", torch.tensor(low).reshape(1, 1, -1))
        self.register_buffer("high", torch.tensor(high).reshape(1, 1, -1))
        self.level = level

    def _step(self, x):
        batch, channels, length = x.shape
        padding = self.low.shape[-1] - 2
        if length % 2:
            x = F.pad(x, (0, 1), mode="replicate")
        x = F.pad(x, (padding, padding), mode="circular") if padding else x
        flat = x.reshape(batch * channels, 1, -1)
        low = F.conv1d(flat, self.low, stride=2).reshape(batch, channels, -1)
        high = F.conv1d(flat, self.high, stride=2).reshape(batch, channels, -1)
        return low, high

    def forward(self, x):
        approximation, details = x, []
        for _ in range(self.level):
            approximation, detail = self._step(approximation)
            details.append(detail)
        return [approximation, *reversed(details)]


class ResolutionMixer(nn.Module):
    """Patch a single resolution, then mix patch and feature axes."""

    def __init__(self, length, patch_len, stride, width, tfactor, dfactor, dropout, horizon):
        super().__init__()
        self.length, self.patch_len, self.stride = length, min(patch_len, length), stride
        self.patch_count = max(1, math.ceil(max(0, length - self.patch_len) / stride) + 1)
        self.embedding = nn.Linear(self.patch_len, width)
        token_hidden = max(self.patch_count, tfactor * self.patch_count)
        feature_hidden = max(width, dfactor * width)
        self.token_mixer = nn.Sequential(
            nn.Linear(self.patch_count, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, self.patch_count),
        )
        self.feature_mixer = nn.Sequential(
            nn.Linear(width, feature_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(feature_hidden, width)
        )
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.forecast = nn.Linear(self.patch_count * width, horizon)

    def forward(self, x):
        needed = (self.patch_count - 1) * self.stride + self.patch_len
        x = F.pad(x, (0, max(0, needed - x.shape[-1])))
        patches = self.embedding(x.unfold(-1, self.patch_len, self.stride)[..., : self.patch_count, :])
        patches = self.norm1(patches + self.token_mixer(patches.transpose(-1, -2)).transpose(-1, -2))
        patches = self.norm2(patches + self.feature_mixer(patches))
        return self.forecast(patches.flatten(-2))


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        d_model=128,
        dropout=0.1,
        tfactor=5,
        dfactor=5,
        wavelet="db2",
        level=1,
        patch_len=16,
        stride=8,
        no_decomposition=False,
    ):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.no_decomposition = no_decomposition
        self.revin = RevIN(enc_in)
        self.wavelet = None if no_decomposition else OrthogonalWaveletAnalysis(wavelet, level)
        lengths = [seq_len] if no_decomposition else self._resolution_lengths(seq_len, wavelet, level)
        self.branches = nn.ModuleList(
            ResolutionMixer(length, patch_len, stride, d_model, tfactor, dfactor, dropout, pred_len)
            for length in lengths
        )
        self.resolution_logits = nn.Parameter(torch.zeros(len(lengths)))
        self.last_resolution_weights = None

    @staticmethod
    def _resolution_lengths(length, wavelet, level):
        filter_len = 2 if wavelet in {"haar", "db1"} else 4
        sizes, current = [], length
        for _ in range(level):
            if current % 2:
                current += 1
            current = (current + 2 * (filter_len - 2) - filter_len) // 2 + 1
            sizes.append(current)
        return [sizes[-1], *reversed(sizes)]

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm").transpose(1, 2)
        resolutions = [normalized] if self.wavelet is None else self.wavelet(normalized)
        forecasts = torch.stack([branch(signal) for branch, signal in zip(self.branches, resolutions)], -1)
        weights = self.resolution_logits.softmax(0)
        self.last_resolution_weights = weights
        output = (forecasts * weights).sum(-1).transpose(1, 2)
        return self.revin(output, "denorm")

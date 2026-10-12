"""Independent TimeKAN implementation from the paper's CFD/M-KAN equations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN


def frequency_upsample(values: torch.Tensor, target_length: int) -> torch.Tensor:
    """IFFT(Padding(FFT(x))) with amplitude preserved across lengths."""
    source_length = values.shape[1]
    spectrum = torch.fft.rfft(values, dim=1)
    target_bins = target_length // 2 + 1
    padded = values.new_zeros(values.shape[0], target_bins, values.shape[2],
                              dtype=spectrum.dtype)
    copied = min(target_bins, spectrum.shape[1])
    padded[:, :copied] = spectrum[:, :copied]
    restored = torch.fft.irfft(padded, n=target_length, dim=1)
    return restored * (target_length / source_length)


class ChebyshevKAN(nn.Module):
    """Equation (7): learned sums of Chebyshev bases over channels."""

    def __init__(self, width: int, order: int) -> None:
        super().__init__()
        self.order = order
        self.coefficients = nn.Parameter(torch.empty(width, width, order + 1))
        nn.init.normal_(self.coefficients, std=(width * (order + 1)) ** -0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        bounded = values.tanh()
        bases = [torch.ones_like(bounded)]
        if self.order >= 1:
            bases.append(bounded)
        for _ in range(2, self.order + 1):
            bases.append(2 * bounded * bases[-1] - bases[-2])
        basis = torch.stack(bases, dim=-1)
        return torch.einsum("btji,oji->bto", basis, self.coefficients)


class MultiOrderKAN(nn.Module):
    def __init__(self, width: int, order: int) -> None:
        super().__init__()
        self.kan = ChebyshevKAN(width, order)
        self.temporal = nn.Conv1d(width, width, kernel_size=3, padding=1,
                                  padding_mode="circular", groups=width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(values.transpose(1, 2)).transpose(1, 2)
        return self.kan(values) + temporal


class TimeKANBlock(nn.Module):
    def __init__(self, width: int, levels: int, begin_order: int) -> None:
        super().__init__()
        self.learners = nn.ModuleList([
            MultiOrderKAN(width, begin_order + levels - index - 1)
            for index in range(levels)
        ])

    def forward(self, hierarchy: list[torch.Tensor]) -> list[torch.Tensor]:
        bands = []
        for upper, lower in zip(hierarchy, hierarchy[1:]):
            bands.append(upper - frequency_upsample(lower, upper.shape[1]))
        bands.append(hierarchy[-1])
        learned = [learner(band) for learner, band in zip(self.learners, bands)]
        mixed = [learned[-1]]
        for band in reversed(learned[:-1]):
            mixed.append(band + frequency_upsample(mixed[-1], band.shape[1]))
        return list(reversed(mixed))


class Model(nn.Module):
    def __init__(
        self, seq_len: int, pred_len: int, label_len: int, features: str,
        enc_in: int, c_out: int | None = None, d_model: int = 16,
        e_layers: int = 1, down_sampling_window: int = 2,
        down_sampling_layers: int = 1, begin_order: int = 0,
        moving_avg: int = 25, dropout: float = 0.1, embed: str = "timeF",
        freq: str = "h", use_norm: int = 1,
    ) -> None:
        super().__init__()
        del label_len, features, moving_avg, dropout, embed, freq
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window = down_sampling_window
        self.levels = down_sampling_layers + 1
        self.normalization = RevIN(enc_in, affine=False, enabled=bool(use_norm))
        self.embedding = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            TimeKANBlock(d_model, self.levels, begin_order) for _ in range(e_layers)
        ])
        self.readout = nn.Linear(d_model, 1)
        self.forecast = nn.Linear(seq_len, pred_len)

    def _hierarchy(self, values: torch.Tensor) -> list[torch.Tensor]:
        levels = [values]
        for _ in range(self.levels - 1):
            pooled = F.avg_pool1d(levels[-1].transpose(1, 2), self.window,
                                  stride=self.window).transpose(1, 2)
            levels.append(pooled)
        return levels

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        normalized = self.normalization(x_enc, "norm")
        batch, _, channels = normalized.shape
        flattened = normalized.transpose(1, 2).reshape(batch * channels, self.seq_len, 1)
        hierarchy = [self.embedding(level) for level in self._hierarchy(flattened)]
        for block in self.blocks:
            hierarchy = block(hierarchy)
        highest = self.readout(hierarchy[0]).squeeze(-1)
        prediction = self.forecast(highest).reshape(batch, channels, self.pred_len).transpose(1, 2)
        return self.normalization(prediction, "denorm")

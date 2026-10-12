"""Independent InterPDN implementation of direct per-step distributions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def _normal_support(size: int, bound: float) -> tuple[torch.Tensor, torch.Tensor]:
    normal = torch.distributions.Normal(0.0, 1.0)
    lo = normal.cdf(torch.tensor(-bound))
    hi = normal.cdf(torch.tensor(bound))
    probability = torch.linspace(float(lo), float(hi), size + 1)
    boundaries = normal.icdf(probability).clamp(-bound, bound)
    first = (boundaries[:-1] + boundaries[1:]) * 0.5
    second = torch.cat(((first[:-1] + first[1:]) * 0.5, first[-1:].new_tensor([bound])))
    return first, second


class DistributionBranch(nn.Module):
    """Channel-independent seasonal/trend backbone and probability head."""

    def __init__(self, seq_len: int, pred_len: int, support_size: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.support_size = support_size
        self.seasonal_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.seasonal_decoder = nn.Linear(seq_len, pred_len)
        self.trend_decoder = nn.Sequential(nn.Linear(seq_len, seq_len), nn.Linear(seq_len, pred_len))
        self.probability_head = nn.Linear(pred_len * 2, pred_len * support_size)

    def forward(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        seasonal = self.seasonal_conv(seasonal.unsqueeze(1)).squeeze(1) + seasonal
        seasonal = F.gelu(self.seasonal_decoder(seasonal))
        trend = self.trend_decoder(trend)
        logits = self.probability_head(torch.cat((seasonal, trend), dim=-1))
        return logits.reshape(-1, self.pred_len, self.support_size)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        support_size: int = 31,
        support_bound: float = 4.0,
        ema_decay: float = 0.8,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.ema_decay = ema_decay
        self.use_revin = use_revin
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        first, second = _normal_support(support_size, support_bound)
        self.register_buffer("support_first", first)
        self.register_buffer("support_second", second)
        self.branches = nn.ModuleList(
            [DistributionBranch(seq_len, pred_len, support_size) for _ in range(2)]
        )
        self.last_probabilities: tuple[torch.Tensor, torch.Tensor] | None = None

    def _ema(self, x: torch.Tensor) -> torch.Tensor:
        values = [x[:, :1]]
        for index in range(1, self.seq_len):
            values.append(self.ema_decay * values[-1] + (1.0 - self.ema_decay) * x[:, index : index + 1])
        return torch.cat(values, dim=1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        if self.use_revin:
            x_enc = self.revin(x_enc, "norm")
        channel_sequences = x_enc.permute(0, 2, 1).reshape(-1, self.seq_len)
        trend = self._ema(channel_sequences)
        seasonal = channel_sequences - trend
        probability_first = self.branches[0](seasonal, trend).softmax(dim=-1)
        probability_second = self.branches[1](seasonal, trend).softmax(dim=-1)
        expectation_first = probability_first @ self.support_first
        expectation_second = probability_second @ self.support_second
        confidence_first = probability_first.amax(dim=-1)
        confidence_second = probability_second.amax(dim=-1)
        weight = confidence_first / (confidence_first + confidence_second).clamp_min(1e-12)
        forecast = weight * expectation_first + (1.0 - weight) * expectation_second
        self.last_probabilities = (probability_first, probability_second)
        output = forecast.reshape(x_enc.shape[0], self.enc_in, self.pred_len).transpose(1, 2)
        if self.use_revin:
            output = self.revin(output, "denorm")
        return output

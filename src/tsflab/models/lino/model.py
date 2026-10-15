"""Paper-driven local implementation of LiNo.

LiNo embeds every variate's whole lookback window as one token (iTransformer
style) and applies Recursive Residual Decomposition (RRD, Eq. 1): each of the
``layers`` LiNo blocks first extracts a linear pattern with a learnable
full-receptive-field autoregressive filter (Li block, Eq. 2), subtracts it,
then extracts a nonlinear pattern (No block: time- and frequency-domain linear
maps, a softmax-weighted channel mean, and MLPs) and subtracts that as well.
Every extracted pattern is projected to the horizon and all projections are
summed (Eq. 4) inside non-affine RevIN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.revin import RevIN


class LinearExtractor(nn.Module):
    """Li block: a per-variate causal AR filter over the full embedding axis.

    Eq. (2): ``L[c, d] = sum_{j <= d} phi[c, .] * H[c, j] + beta[c]`` realised
    (as the paper states) by a depthwise convolution whose kernel spans the
    whole axis, so coefficients depend on the lag ``d - j``; the input is
    left-padded with ``D - 1`` zeros so the output keeps length ``D`` and each
    position sees every earlier one. Coefficients start as a uniform average
    (softmax of ones) and the bias at zero.
    """

    def __init__(self, channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.full((channels, 1, d_model), 1.0 / d_model))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, C, D]
        padded = F.pad(hidden, (self.d_model - 1, 0))
        linear = F.conv1d(padded, self.weight, self.bias, groups=hidden.shape[1])
        return self.dropout(linear)


class FrequencyLinear(nn.Module):
    """Complex linear map between the rFFT bins of the feature axis (FITS-style)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        bins = d_model // 2 + 1
        real = nn.Linear(bins, bins)  # real default init; imaginary parts start at zero
        self.weight = nn.Parameter(real.weight.detach().to(torch.cfloat))
        self.bias = nn.Parameter(real.bias.detach().to(torch.cfloat))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(hidden, dim=-1)
        mapped = spectrum @ self.weight.transpose(0, 1) + self.bias
        return torch.fft.irfft(mapped, n=self.d_model, dim=-1)


def softmax_channel_mean(features: torch.Tensor) -> torch.Tensor:
    """Softmax over the variate axis, weighted sum, repeated back to every variate."""
    weights = torch.softmax(features, dim=1)
    pooled = (features * weights).sum(dim=1, keepdim=True)
    return pooled.expand_as(features)


def _mlp(d_in: int, d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_model),
        nn.Tanh(),
        nn.Dropout(dropout),
        nn.Linear(d_model, d_model),
    )


class NonlinearExtractor(nn.Module):
    """No block: temporal + frequency projection, channel mixing, and an MLP."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Linear(d_model, d_model)
        self.frequency = FrequencyLinear(d_model)
        self.channel_mlp = _mlp(2 * d_model, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = _mlp(d_model, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def time_frequency(self, residual: torch.Tensor) -> torch.Tensor:
        # N^TF = Tanh(N^T + N^F)
        return torch.tanh(self.temporal(residual) + self.frequency(residual))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        tf = self.time_frequency(residual)
        # Inter-series dependencies: the softmax-weighted channel mean of N^TF is
        # concatenated with the block input R^L (as the official code does) and
        # passed through an MLP to give N^C.
        channel = self.channel_mlp(torch.cat([residual, softmax_channel_mean(tf)], dim=-1))
        tfc = self.norm1(tf + channel)
        return self.norm2(tfc + self.mlp(tfc))


class LiNoBlock(nn.Module):
    """One RRD level: R^L = H - Li, R^N = R^L - No, with one horizon head per pattern."""

    def __init__(self, enc_in: int, d_model: int, pred_len: int, dropout: float) -> None:
        super().__init__()
        self.linear_extractor = LinearExtractor(enc_in, d_model, dropout)
        self.nonlinear_extractor = NonlinearExtractor(d_model, dropout)
        self.linear_head = nn.Linear(d_model, pred_len)
        self.nonlinear_head = nn.Linear(d_model, pred_len)
        with torch.no_grad():
            self.linear_head.weight.fill_(1.0 / d_model)
            self.nonlinear_head.weight.fill_(1.0 / d_model)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        linear = self.linear_extractor(hidden)
        residual = hidden - linear
        nonlinear = self.nonlinear_extractor(residual)
        residual = residual - nonlinear
        return self.linear_head(linear), self.nonlinear_head(nonlinear), residual


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, eps=1e-5, affine=False)
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.blocks = nn.ModuleList(LiNoBlock(enc_in, d_model, pred_len, dropout) for _ in range(layers))

    def block_predictions(self, normalized: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-level (linear, nonlinear) forecasts in normalized space, each [B, C, F]."""
        hidden = self.embedding(normalized, None)
        outputs = []
        for block in self.blocks:
            linear, nonlinear, hidden = block(hidden)
            outputs.append((linear, nonlinear))
        return outputs

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        normalized = self.revin(x_enc, "norm")
        # Eq. (4): sum of every level's linear and nonlinear predictions.
        prediction = sum(lin + non for lin, non in self.block_predictions(normalized))
        return self.revin(prediction.transpose(1, 2), "denorm")

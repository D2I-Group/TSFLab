"""PWS model implementation."""

from __future__ import annotations

import math

import einops
import torch
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def make_act(name: str):
    name = (name or "").lower()
    return _ACTIVATIONS[name]()


def _check_patching(period: int, seq_len: int) -> None:
    if period > seq_len:
        raise ValueError("period must be in [1, seq_len]")


def _parse_hidden_string(analysis_hidden: str) -> list[int]:
    s = analysis_hidden.strip()
    return [int(x) for x in s.split(",") if x] if s else []


def _parse_hidden(analysis_hidden) -> list[int]:
    if isinstance(analysis_hidden, str):
        hidden_sizes = _parse_hidden_string(analysis_hidden)
    elif isinstance(analysis_hidden, (list, tuple)):
        hidden_sizes = [int(x) for x in analysis_hidden]
    else:
        hidden_sizes = []
    return hidden_sizes


class PWSModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        period: int,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        analysis_act: str = "silu",
        analysis_hidden: list[int] | tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        _check_patching(period, seq_len)
        self.c_in = c_in
        self.period = period
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.num_seq_periods = self.seq_len // self.period
        self.num_future_periods = (self.pred_len + self.period - 1) // self.period
        self.num_patches = math.ceil(self.period / self.patch_size)

        self.hidden_sizes = _parse_hidden(analysis_hidden)
        self.analysis_act = analysis_act

        self.analysis_layers = nn.ModuleList(
            [self.make_analysis() for _ in range(self.num_patches)]
        )
        self.weighted_sum_layers = nn.ModuleList(
            [self.make_wsum() for _ in range(self.num_patches)]
        )

    def make_analysis(self):
        act = make_act(self.analysis_act)
        sizes = [self.num_seq_periods] + self.hidden_sizes + [self.num_seq_periods]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 and act is not None:
                layers.append(act)
        return nn.Sequential(*layers)

    def make_wsum(self):
        return nn.Linear(self.num_seq_periods, self.num_future_periods)

    def _check_input(self, x: torch.Tensor) -> None:
        check_window(x, self.seq_len, self.c_in)

    def _to_periods(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_len % self.period != 0:
            x = x[:, :, -(self.num_seq_periods * self.period) :]

        x = einops.rearrange(
            x,
            "b n (i p) -> (b n) p i",
            p=int(self.period),
        )
        _, period, num_seq_periods = x.shape
        if period != self.period or num_seq_periods != self.num_seq_periods:
            raise ValueError("Unexpected period dimensions for PWS")
        return x

    def _patch_forecast(self, x: torch.Tensor) -> torch.Tensor:
        bn = x.shape[0]
        out = x.new_zeros(bn, self.period, self.num_future_periods)

        for patch_idx in range(self.num_patches):
            start = patch_idx * self.patch_size
            end = min((patch_idx + 1) * self.patch_size, self.period)
            if start >= end:
                continue
            x_patch = x[:, start:end, :]
            x_patch = self.analysis_layers[patch_idx](x_patch) + x_patch
            x_patch = self.weighted_sum_layers[patch_idx](x_patch)
            out[:, start:end, :] = x_patch
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        if self.revin:
            x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)
        x = self._to_periods(x)

        x = self._patch_forecast(x)
        x = einops.rearrange(
            x,
            "(b n) p i -> b n (i p)",
            n=self.c_in,
        )
        x = x[:, :, : self.pred_len]
        x = x.permute(0, 2, 1)

        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        period: int,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        revin: bool,
        affine: bool,
        subtract_last: bool,
        analysis_act: str,
        analysis_hidden: list[int],
    ):
        super().__init__()
        self.model = PWSModel(
            c_in=c_in,
            period=period,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_size=patch_size,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            analysis_act=analysis_act,
            analysis_hidden=analysis_hidden,
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        return self.model(x_enc)

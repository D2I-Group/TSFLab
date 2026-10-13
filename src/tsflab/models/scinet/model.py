"""Clean-room SCINet implementation following Eqs. (1)--(2) and the SCI-Tree."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def interleave(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """Inverse of odd/even splitting, including an unmatched final even item."""
    batch, even_length, channels = even.shape
    odd_length = odd.shape[1]
    output = even.new_empty(batch, even_length + odd_length, channels)
    output[:, 0::2] = even
    output[:, 1::2] = odd
    return output


class TemporalOperator(nn.Module):
    """Paper appendix's two normal 1-D convolutions with replication padding."""

    def __init__(self, channels, hidden_size, kernel_size, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.first = nn.Conv1d(channels, hidden_size, kernel_size)
        self.second = nn.Conv1d(hidden_size, channels, 3)
        self.dropout = nn.Dropout(dropout)

    def _same_pad(self, values):
        total = self.kernel_size + 1
        return F.pad(values, (total // 2, total - total // 2), mode="replicate")

    def forward(self, values):
        hidden = self.first(self._same_pad(values))
        hidden = self.dropout(F.leaky_relu(hidden, negative_slope=0.01))
        return torch.tanh(self.second(hidden))


class SCIInteraction(nn.Module):
    """Feven/Fodd multiplicative scaling followed by additive/subtractive coupling."""

    def __init__(self, channels, hidden_size, kernel_size, dropout):
        super().__init__()
        self.phi = TemporalOperator(channels, hidden_size, kernel_size, dropout)
        self.psi = TemporalOperator(channels, hidden_size, kernel_size, dropout)
        self.rho = TemporalOperator(channels, hidden_size, kernel_size, dropout)
        self.eta = TemporalOperator(channels, hidden_size, kernel_size, dropout)

    def forward(self, values):
        even = values[:, 0::2].transpose(1, 2)
        odd = values[:, 1::2].transpose(1, 2)
        if even.shape[-1] != odd.shape[-1]:
            odd = F.pad(odd, (0, 1), mode="replicate")
            trim_odd = True
        else:
            trim_odd = False
        scaled_odd = odd * torch.exp(self.phi(even).clamp(-8, 8))
        scaled_even = even * torch.exp(self.psi(odd).clamp(-8, 8))
        updated_odd = scaled_odd - self.rho(scaled_even)
        updated_even = scaled_even + self.eta(scaled_odd)
        if trim_odd:
            updated_odd = updated_odd[..., :-1]
        return updated_even.transpose(1, 2), updated_odd.transpose(1, 2)


class SCITree(nn.Module):
    """Recursive multi-resolution arrangement of 2**levels-1 SCI blocks."""

    def __init__(self, channels, levels, hidden_size, kernel_size, dropout):
        super().__init__()
        self.levels = levels
        self.interaction = SCIInteraction(channels, hidden_size, kernel_size, dropout)
        if levels > 1:
            self.even_tree = SCITree(
                channels, levels - 1, hidden_size, kernel_size, dropout
            )
            self.odd_tree = SCITree(
                channels, levels - 1, hidden_size, kernel_size, dropout
            )

    def forward(self, values):
        even, odd = self.interaction(values)
        if self.levels > 1:
            even, odd = self.even_tree(even), self.odd_tree(odd)
        return interleave(even, odd)


class SCINetStack(nn.Module):
    def __init__(
        self, seq_len, pred_len, channels, levels, hidden_size, kernel_size, dropout
    ):
        super().__init__()
        self.tree = SCITree(channels, levels, hidden_size, kernel_size, dropout)
        self.forecast = nn.Linear(seq_len, pred_len, bias=False)

    def forward(self, values):
        enhanced = values + self.tree(values)
        return self.forecast(enhanced.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    """Original single/two-stack forecast path with optional PE and RIN."""

    def __init__(self, seq_len, pred_len, enc_in, num_stacks=1, num_levels=3,
                 hidden_size=None, kernel_size=5, dropout=0.0,
                 positional_encoding=False, instance_norm=False):
        super().__init__()
        hidden_size = enc_in if hidden_size is None else hidden_size
        if min(seq_len, pred_len, enc_in, num_levels, hidden_size, kernel_size) < 1:
            raise ValueError("all dimensions must be positive")
        if num_stacks not in (1, 2):
            raise ValueError("the official forecast implementation supports one or two stacks")
        if seq_len % (2 ** num_levels) or (num_stacks == 2 and (seq_len + pred_len) % (2 ** num_levels)):
            raise ValueError("seq_len must be divisible by 2**num_levels")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.positional_encoding, self.instance_norm = positional_encoding, instance_norm
        self.stacks = nn.ModuleList([SCINetStack(
            seq_len if i == 0 else seq_len + pred_len, pred_len, enc_in,
            num_levels, hidden_size, kernel_size, dropout) for i in range(num_stacks)])
        pe_width = enc_in + enc_in % 2
        frequency = torch.exp(torch.arange(pe_width // 2) * (-math.log(10000.0) / max(pe_width // 2 - 1, 1)))
        position = torch.arange(seq_len).unsqueeze(1)
        self.register_buffer("position_encoding", torch.cat((torch.sin(position * frequency), torch.cos(position * frequency)), -1)[:, :enc_in])
        if instance_norm:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, enc_in))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch,{self.seq_len},{self.enc_in}]")
        history = x_enc
        if self.positional_encoding:
            history = history + self.position_encoding
        if self.instance_norm:
            center = history.mean(1, keepdim=True).detach()
            scale = history.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
            history = (history - center) / scale * self.affine_weight + self.affine_bias
        prediction = self.stacks[0](history)
        if len(self.stacks) == 2:
            prediction = self.stacks[1](torch.cat((history, prediction), dim=1))
        if self.instance_norm:
            prediction = ((prediction - self.affine_bias) / (self.affine_weight + 1e-10)) * scale + center
        return prediction

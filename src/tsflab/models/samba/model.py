"""SAMBA / SDE-Mamba: simplified Mamba with disentangled time and variate encoders.

The series is instance-normalized, cut into overlapping patches and embedded.  Two
stacks of S-Mamba blocks then run in parallel on the same embedding (paper
Sec. 5.2): a cross-time stack over patches of each variate (with a learnable
position term) and a cross-variate stack over variates of each patch.  Their outputs
are concatenated, mixed by an FFN, flattened and projected to the horizon.  The
"simplification" (Sec. 5.1) removes the activation between the depthwise convolution
and the selective SSM; the SiLU gate is kept.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class SimplifiedMambaMixer(MambaBlock):
    """Mamba mixer whose convolution output feeds the SSM without an activation."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        use_act: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_floor: float = 1e-4,
    ) -> None:
        dt_rank = math.ceil(d_model / 16)
        super().__init__(d_model, expand * d_model, dt_rank, d_conv, d_state)
        self.use_act = use_act
        # Reference Mamba step-size initialization (softplus(bias) in [dt_min, dt_max]).
        nn.init.uniform_(self.dt_proj.weight, -(dt_rank**-0.5), dt_rank**-0.5)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        signal, gate = self.in_proj(x).split([self.d_inner, self.d_inner], dim=-1)
        signal = self.conv1d(signal.transpose(1, 2))[:, :, :length].transpose(1, 2)
        if self.use_act:
            signal = F.silu(signal)
        return self.out_proj(self.ssm(signal) * F.silu(gate))


class GatedMLP(nn.Module):
    """SiLU-gated feed-forward with hidden width rounded up to a multiple of 128."""

    def __init__(self, d_model: int, d_ff: int, multiple_of: int = 128) -> None:
        super().__init__()
        hidden = (d_ff + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(value * F.silu(gate))


class SimplifiedMambaBlock(nn.Module):
    """Add -> LayerNorm -> mixer, then add -> LayerNorm -> gated MLP; returns (hidden, residual)."""

    def __init__(
        self, d_model: int, d_state: int, d_ff: int, use_act: bool
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = SimplifiedMambaMixer(d_model, d_state, use_act=use_act)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff)

    def forward(self, hidden, residual=None):
        residual = hidden if residual is None else hidden + residual
        hidden = self.mixer(self.norm(residual))
        residual = hidden + residual
        return self.mlp(self.norm2(residual)), residual


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_ff: int = 128,
        e_layers: int = 1,
        d_layers: int = 1,
        d_state1: int = 16,
        d_state2: int = 16,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        use_act: bool = False,
    ) -> None:
        super().__init__()
        if patch_len > seq_len + stride:
            raise ValueError("patch_len exceeds the padded history")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len + stride - patch_len) // stride + 1
        self.revin = RevIN(enc_in, affine=False)
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.patch_projection = nn.Linear(patch_len, d_model)
        # Learnable position term of shape [patches, 1] broadcast over d_model.
        self.position = nn.Parameter(torch.empty(self.patch_num, 1).uniform_(-0.02, 0.02))
        self.encoder_time = nn.ModuleList(
            SimplifiedMambaBlock(d_model, d_state1, d_ff, use_act) for _ in range(e_layers)
        )
        self.encoder_var = nn.ModuleList(
            SimplifiedMambaBlock(d_model, d_state2, d_ff, use_act) for _ in range(d_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.norm_var = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.head = FlattenForecastHead(False, enc_in, d_model * self.patch_num, pred_len)

    @staticmethod
    def _run_stack(blocks, norm, hidden):
        """Official residual-stream convention: the shared norm re-normalizes the residual."""
        residual = None
        for block in blocks:
            hidden, residual = block(hidden, residual)
            residual = norm(residual)
        return norm(hidden + residual)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch, channels = x_enc.shape[0], self.enc_in
        series = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        patches = self.padding_patch_layer(series).unfold(-1, self.patch_len, self.stride)
        embedded = self.patch_projection(patches)  # [B, C, J, D]
        j, d = self.patch_num, self.d_model

        branches = []
        if self.encoder_time:
            time_in = embedded.reshape(batch * channels, j, d) + self.position
            time_out = self._run_stack(self.encoder_time, self.norm, time_in)
            branches.append(time_out.reshape(batch, channels, j, d))
        if self.encoder_var:
            var_in = embedded.permute(0, 2, 1, 3).reshape(batch * j, channels, d)
            var_out = self._run_stack(self.encoder_var, self.norm_var, var_in)
            branches.append(var_out.reshape(batch, j, channels, d).permute(0, 2, 1, 3))
        if len(branches) == 1:
            fused = branches[0]
        else:
            fused = self.fusion(torch.cat(branches, dim=-1))
        forecast = self.head(fused.transpose(2, 3)).transpose(1, 2)  # [B, pred_len, C]
        return self.revin(forecast, "denorm")

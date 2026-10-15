"""MTS-UNMixers: channel-time dual unmixing with Mamba encoders (Zhu et al., 2024).

The window is unmixed twice (Sec. III-B, Eqs. 6-9, 17-18):

* Time path: a patch Mamba encoder turns each channel into softmax mixing
  coefficients ``S_c`` (``k2`` per channel, Eq. 15) over learnable temporal bases
  ``A_c`` (history, ``T x k2``) and ``A_p`` (future, ``H x k2``); the same
  ``S_c`` reconstructs the history and predicts the future (Eq. 17).
* Channel path: a bidirectional Mamba over channel tokens produces per-sample
  channel bases ``A_t`` (``N x k1``, Eq. 16), mixed by learnable softmax
  coefficients ``S_t`` (history, ``k1 x T``) and ``S_p`` (future, ``k1 x H``)
  that are shared by every sample (Eq. 18).

Both paths are concatenated and projected (Eqs. 19-20); training adds the L1
reconstruction of the history to the forecasting loss (Eq. 21).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.mamba import MambaBlock
from tsflab.models._inputs import check_window


class TimeEncoderBlock(nn.Module):
    """Fig. 3 / Eq. (14): ``Norm -> Mamba`` over the patch tokens of one channel."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = MambaBlock(d_model, expand * d_model, math.ceil(d_model / 16), d_conv, d_state)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(tokens))


class DirectionalScan(nn.Module):
    """One branch of Fig. 4: causal depthwise conv, SiLU and a selective SSM over the token axis."""

    def __init__(self, d_inner: int, d_state: int, dt_rank: int, d_conv: int) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.dt_rank, self.d_state = dt_rank, d_state

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        length = signal.shape[1]
        signal = F.silu(self.conv1d(signal.transpose(1, 2))[..., :length].transpose(1, 2))
        delta, b, c = self.x_proj(signal).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return MambaBlock.selective_scan(signal, delta, -torch.exp(self.A_log.float()), b, c, self.D.float())


class BiMambaBlock(nn.Module):
    """Eq. (16) / Fig. 4: shared input projection, forward and backward scans gated by ``ReLU``, summed."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        d_inner, dt_rank = expand * d_model, math.ceil(d_model / 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.forward_scan = DirectionalScan(d_inner, d_state, dt_rank, d_conv)
        self.backward_scan = DirectionalScan(d_inner, d_state, dt_rank, d_conv)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        signal, gate = self.in_proj(self.norm(tokens)).chunk(2, dim=-1)
        gate = F.relu(gate)
        forward = self.forward_scan(signal) * gate
        backward = self.backward_scan(signal.flip(1)).flip(1) * gate
        return self.out_proj(forward + backward)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        e_layers: int = 1,
        patch_len: int = 16,
        time_bases: int = 16,
        channel_bases: int = 16,
        dropout: float = 0.1,
        recon_weight: float = 1.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len = patch_len
        self.num_patches = math.ceil(seq_len / patch_len)
        self.recon_weight = recon_weight
        self.use_norm = use_norm
        self.dropout = nn.Dropout(dropout)
        # Time unmixing: patch tokens -> Mamba -> Linear -> softmax gives S_c (Eqs. 14-15).
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.time_encoder = nn.ModuleList(TimeEncoderBlock(d_model, d_state, d_conv, expand) for _ in range(e_layers))
        self.coefficient_head = nn.Linear(self.num_patches * d_model, channel_bases)
        self.history_bases = nn.Parameter(torch.empty(seq_len, channel_bases))  # A_c
        self.future_bases = nn.Parameter(torch.empty(pred_len, channel_bases))  # A_p
        # Channel unmixing: channel tokens -> Bi-Mamba -> Linear gives A_t (Eq. 16).
        self.channel_embedding = nn.Linear(seq_len, d_model)
        self.channel_encoder = nn.ModuleList(BiMambaBlock(d_model, d_state, d_conv, expand) for _ in range(e_layers))
        self.basis_head = nn.Linear(d_model, time_bases)
        self.history_logits = nn.Parameter(torch.zeros(time_bases, seq_len))  # S_t before softmax
        self.future_logits = nn.Parameter(torch.zeros(time_bases, pred_len))  # S_p before softmax
        # Concat & projection (Eqs. 19-20).
        self.forecast_projection = nn.Linear(2 * pred_len, pred_len)
        self.history_projection = nn.Linear(2 * seq_len, seq_len)
        nn.init.xavier_uniform_(self.history_bases)
        nn.init.xavier_uniform_(self.future_bases)

    def channel_coefficients(self, series: torch.Tensor) -> torch.Tensor:
        """``[B, N, T] -> S_c^T [B, N, k2]``: per-channel softmax coefficients (Eqs. 14-15)."""
        batch, channels, length = series.shape
        pad = self.num_patches * self.patch_len - length
        if pad:
            series = torch.cat((series, series[..., -1:].expand(-1, -1, pad)), dim=-1)
        patches = series.reshape(batch * channels, self.num_patches, self.patch_len)
        tokens = self.dropout(self.patch_embedding(patches))
        for block in self.time_encoder:
            tokens = block(tokens)
        logits = self.coefficient_head(tokens.reshape(batch, channels, -1))
        return torch.softmax(logits, dim=-1)

    def channel_bases(self, series: torch.Tensor) -> torch.Tensor:
        """``[B, N, T] -> A_t [B, N, k1]``: per-sample bases from the bidirectional encoder (Eq. 16)."""
        tokens = self.dropout(self.channel_embedding(series))
        for block in self.channel_encoder:
            tokens = block(tokens)
        return self.basis_head(tokens)

    def unmix(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruction ``[B, N, T]`` and forecast ``[B, N, H]`` of standardized ``[B, N, T]`` input."""
        coefficients = self.channel_coefficients(series)  # S_c, shared by history and future
        channel_history = coefficients @ self.history_bases.T  # X'_c = A_c S_c (Eq. 17)
        channel_future = coefficients @ self.future_bases.T  # X^_c = A_p S_c
        bases = self.channel_bases(series)  # A_t, shared by history and future
        time_history = bases @ torch.softmax(self.history_logits, dim=0)  # X'_t = A_t S_t (Eq. 18)
        time_future = bases @ torch.softmax(self.future_logits, dim=0)  # X^_t = A_t S_p
        history = self.history_projection(torch.cat((channel_history, time_history), dim=-1))  # Eq. (20)
        future = self.forecast_projection(torch.cat((channel_future, time_future), dim=-1))  # Eq. (19)
        return history, future

    def forecast_and_reconstruct(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the forecast ``[B, H, N]`` and the reconstructed history ``[B, T, N]``."""
        check_window(x_enc, self.seq_len, self.enc_in)
        series = x_enc.transpose(1, 2)
        if self.use_norm:
            mean = series.mean(-1, keepdim=True).detach()
            scale = series.var(-1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
            series = (series - mean) / scale
        history, future = self.unmix(series)
        if self.use_norm:
            history, future = history * scale + mean, future * scale + mean
        return future.transpose(1, 2), history.transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast_and_reconstruct(x_enc)[0]

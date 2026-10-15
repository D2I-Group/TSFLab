"""Clean-room local implementation of TimePro's variable- and time-aware hyper-state.

Each channel is patched independently and embedded (as in PatchTST), then the
concatenated per-channel patch embeddings become one token per *variate* fed
to a stack of bidirectional selective-state-space blocks (paper Section 3).
Inside each block (:class:`ProMamba`), the scalar-state selective-scan hidden
state is reshaped into a ``(time-patch, variate)`` grid and locally mixed
before being read out -- the paper's "hyper-state": a state that is aware of
both the variate axis (the scan's own sequence axis) and the time-patch axis
(folded into the channel axis by the patch embedding). The official
implementation mixes this grid with a CUDA-only deformable convolution
(DCNv4); this implementation reuses the portable
``tsflab.models._components.hyper_state_scan.GridStateMixer`` depthwise convolution
instead, trading learned sampling offsets for a fixed local receptive field
so the model runs on CPU without a custom kernel (documented in the model
card).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.hyper_state_scan import GridStateMixer, diagonal_selective_scan
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._inputs import check_window


class PatchEmbedding(nn.Module):
    """Per-channel overlapping patches, linearly embedded (PatchTST-style)."""

    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.padding = nn.ReplicationPad1d((0, stride))
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """``x``: ``(batch, channels, seq_len)`` -> ``(batch * channels, patch_num, d_model)``."""
        n_vars = x.shape[1]
        patches = self.padding(x).unfold(-1, self.patch_len, self.stride)
        tokens = patches.reshape(patches.shape[0] * patches.shape[1], patches.shape[2], patches.shape[3])
        return self.dropout(self.projection(tokens)), n_vars


class ProMamba(nn.Module):
    """Selective-scan mixer whose hidden state is spatially mixed on a hyper-state grid."""

    def __init__(self, d_model: int, patch_num: int, d_state: int, d_conv: int, expand: float, dropout: float) -> None:
        super().__init__()
        self.d_inner = int(expand * d_model)
        if self.d_inner % patch_num:
            raise ValueError("expand * d_model must be an integer multiple of patch_num")
        self.per_patch_dim = self.d_inner // patch_num
        self.patch_num = patch_num
        self.dt_rank = max(1, d_model // 16)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2, groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.a_log = nn.Parameter(torch.zeros(self.d_inner))
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.state_mixer = GridStateMixer(self.per_patch_dim)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, n_vars, d_model)`` -> same shape."""
        batch, n_vars, _ = x.shape
        xz = self.in_proj(x)
        inner, gate = xz.chunk(2, dim=-1)
        inner = inner.transpose(1, 2)  # (batch, d_inner, n_vars)
        inner = F.silu(self.conv(inner)[..., :n_vars])

        projected = self.x_proj(inner.transpose(1, 2))  # (batch, n_vars, dt_rank + 2*d_state)
        delta_raw, b, c = projected.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw)).transpose(1, 2)  # (batch, d_inner, n_vars)
        a = -torch.exp(self.a_log)  # scalar-state decay per channel: (d_inner,)
        b_gate = b[..., 0].unsqueeze(1).expand(-1, self.d_inner, -1)  # (batch, d_inner, n_vars)
        c_read = c[..., 0].unsqueeze(1)  # (batch, 1, n_vars)

        state = diagonal_selective_scan(inner, delta, a, b_gate)  # (batch, d_inner, n_vars)

        grid = state.reshape(batch, self.patch_num, self.per_patch_dim, n_vars).permute(0, 2, 1, 3)
        grid = self.state_mixer(grid)
        state = grid.permute(0, 2, 1, 3).reshape(batch, self.d_inner, n_vars)

        y = state * c_read + inner * self.d_skip.view(1, -1, 1)
        y = self.out_norm(y.transpose(1, 2))
        y = y * F.silu(gate)
        return self.out_proj(self.dropout(y))


class ProBlock(nn.Module):
    """Bidirectional hyper-state mixer plus a position-wise MLP, each residual."""

    def __init__(
        self,
        d_model: int,
        patch_num: int,
        d_state: int,
        d_conv: int,
        expand: float,
        dropout: float,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.forward_scan = ProMamba(d_model, patch_num, d_state, d_conv, expand, dropout)
        self.backward_scan = ProMamba(d_model, patch_num, d_state, d_conv, expand, dropout)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward = self.forward_scan(x)
        backward = self.backward_scan(x.flip(dims=[1])).flip(dims=[1])
        x = x + forward + backward
        return x + self.mlp(x)


class Model(nn.Module):
    """Patch-then-variate-scan forecaster with a hyper-state read-out."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 16,
        e_layers: int = 2,
        d_state: int = 1,
        d_conv: int = 5,
        expand: float = 1.0,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.use_norm = use_norm

        self.patch_num = (seq_len + stride - patch_len) // stride + 1
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        self.hidden_dim = d_model * self.patch_num

        self.blocks = nn.ModuleList(
            ProBlock(self.hidden_dim, self.patch_num, d_state, d_conv, expand, dropout) for _ in range(e_layers)
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.projector = nn.Linear(self.hidden_dim, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)

        if self.use_norm:
            x_enc, mean, stdev = standardize(x_enc)

        tokens, n_vars = self.patch_embedding(x_enc.transpose(1, 2))
        enc_out = tokens.reshape(-1, n_vars, self.patch_num * tokens.shape[-1])

        for block in self.blocks:
            enc_out = block(enc_out)
        enc_out = self.final_norm(enc_out)

        forecast = self.projector(enc_out).transpose(1, 2)
        if self.use_norm:
            forecast = destandardize(forecast, mean, stdev)
        return forecast

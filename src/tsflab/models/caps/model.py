"""CAPS: Clock-weighted Aggregation with Prefix-products and Softmax (Pati et al., 2026).

Each channel is centered on its last value and extended to ``L + H`` steps by a
learned linear fill (Eq. 18). Every step gets a dual token: a channel token mixing
all channels and a per-channel value token (Sec. A.3). A stack of pre-RMSNorm
Transformer blocks with CAPS linear attention (Eqs. 7-17) processes each channel's
token sequence independently, and the value slice of the horizon tokens is decoded
back to a scalar with the transposed value embedding.

CAPS attention rotates queries and keys with RoPE (Eq. 8) and scores them through
three additive paths sharing one learned Clock ``Delta``: a Clock-weighted Riemann
softmax (Eqs. 9-11), a diagonal prefix-product decay (Eqs. 12-14) and a Clock
baseline (Eq. 15), concatenated into one causal linear attention (Eqs. 16-17).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.last_value_center import center_on_last_value, restore_last_value
from tsflab.models._components.mamba import RMSNorm

CLOCK_EPS = 1e-6  # epsilon of Eq. (7)
LOG_GATE_FLOOR = -50.0  # numerical floor on the cumulative log-gate of Eq. (12)


def rope_rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Eq. (8): rotate consecutive pairs ``(x[2l], x[2l+1])`` of ``[B, T, H, d]`` by ``t * w_l``."""
    positions = torch.arange(x.shape[1], device=x.device, dtype=freqs.dtype)
    angle = (positions[:, None] * freqs[None, :]).to(x.dtype)[None, :, None, :]  # [1, T, 1, d/2]
    cos, sin = angle.cos(), angle.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def causal_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Eq. (17) with identity ``phi``: ``o_t = sum_{i <= t} (q_t . k_i) v_i`` over ``[B, T, H, d]``."""
    scores = torch.einsum("bthd,bihd->bhti", q, k).tril()
    return torch.einsum("bhti,bihd->bthd", scores, v)


class CAPSAttention(nn.Module):
    """Three-path CAPS linear attention over ``[B, T, D]`` with ``H`` heads."""

    def __init__(self, width: int, n_heads: int, dropout: float, learn_rope_freq: bool) -> None:
        super().__init__()
        self.n_heads = n_heads
        head = width // n_heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.score_gate = nn.Linear(width, width, bias=False)  # W_p (Path 1)
        self.decay_gate = nn.Linear(width, width, bias=False)  # W_g (Path 2)
        self.clock = nn.Linear(width, n_heads, bias=False)  # W_c (Eq. 7)
        self.c_proj = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        freqs = 10000.0 ** (-torch.arange(0, head, 2, dtype=torch.float32) / head)
        if learn_rope_freq:  # Eq. (8): learned frequencies, initialized at the RoPE schedule
            self.rope_freq = nn.Parameter(freqs)
        else:
            self.register_buffer("rope_freq", freqs)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:2], self.n_heads, -1)

    def clock_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (7): ``Delta_t = softplus(W_c h_t) + eps``, one value per head, ``[B, T, H, 1]``."""
        return (F.softplus(self.clock(x)) + CLOCK_EPS).unsqueeze(-1)

    def path_queries_keys(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenated ``(q_cat, k_cat)`` of Eq. (16), each ``[B, T, H, 3 d]``."""
        q = rope_rotate(self.split(self.query(x)), self.rope_freq)
        k = rope_rotate(self.split(self.key(x)), self.rope_freq)
        clock = self.clock_weights(x)

        # Path 1, Eqs. (9)-(11): G_{t,i} = e^{p_i} D_i / sum_{j<=t} e^{p_j} D_j per feature.
        scores = self.split(self.score_gate(x)) + clock.log()
        weight = torch.exp(scores - scores.amax(dim=1, keepdim=True))  # the shift cancels in the ratio
        q1 = q / weight.cumsum(dim=1).clamp_min(torch.finfo(weight.dtype).tiny)
        k1 = k * weight

        # Path 2, Eqs. (12)-(14): A_{t,i} = Gamma_t / Gamma_i = exp(sum_{j=i+1..t} g~_j), g~ = -softplus(g) D.
        log_gamma = (-F.softplus(self.split(self.decay_gate(x))) * clock).cumsum(dim=1)
        log_gamma = log_gamma.clamp_min(LOG_GATE_FLOOR)
        q2 = q * log_gamma.exp()
        k2 = k * (-log_gamma).exp()

        # Path 3, Eq. (15): Clock baseline B_{t,i} = D_i / sum_{j<=t} D_j.
        q3 = q / clock.cumsum(dim=1)
        k3 = k * clock
        return torch.cat((q1, q2, q3), dim=-1), torch.cat((k1, k2, k3), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_cat, k_cat = self.path_queries_keys(x)
        out = causal_linear_attention(q_cat, k_cat, self.split(self.value(x)))
        return self.dropout(self.c_proj(out.reshape(x.shape)))


class FeedForward(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, 4 * width, bias=False)
        self.c_proj = nn.Linear(4 * width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class CAPSBlock(nn.Module):
    """Pre-norm block (Fig. 2b): ``x + Attn(RMSNorm(x))`` then ``x + FFN(RMSNorm(x))``."""

    def __init__(self, width: int, n_heads: int, dropout: float, learn_rope_freq: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.attn = CAPSAttention(width, n_heads, dropout, learn_rope_freq)
        self.ffn_norm = RMSNorm(width)
        self.ffn = FeedForward(width, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


def random_ratio_channel_dropout(x: torch.Tensor) -> torch.Tensor:
    """Sec. A.3 on ``[B, C, T]``: drop channels with a per-sample rate ``r ~ U(0, 1)``, rescale survivors."""
    rate = torch.rand(x.shape[0], 1, device=x.device)
    keep = (torch.rand(x.shape[0], x.shape[1], device=x.device) > rate).to(x.dtype)
    keep = keep / keep.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return x * keep.unsqueeze(-1)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 16,
        value_dim: int = 16,
        n_heads: int = 4,
        e_layers: int = 3,
        dropout: float = 0.0,
        channel_dropout: bool = True,
        learn_rope_freq: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.value_dim = value_dim
        self.channel_dropout = channel_dropout
        width = d_model + value_dim
        self.channel_token = nn.Linear(enc_in, d_model, bias=False)  # c_t = W_c x_t
        self.value_embedding = nn.Parameter(0.02 * torch.randn(enc_in, value_dim))  # V_c
        self.position = nn.Parameter(torch.zeros(seq_len + pred_len, width))
        self.channel_embedding = nn.Parameter(torch.zeros(enc_in, width))
        self.blocks = nn.ModuleList(
            CAPSBlock(width, n_heads, dropout, learn_rope_freq) for _ in range(e_layers)
        )
        self.apply(self._init_linear)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):  # GPT-2 residual-projection scaling (Sec. A.2)
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * e_layers))
        self.horizon_fill = nn.Linear(seq_len, pred_len, bias=False)  # W_ext of Eq. (18)

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def extend(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Last-value centering and Eq. (18): ``[B, L, C] -> ([B, C, L + H], level [B, 1, C])``."""
        centered, level = center_on_last_value(x_enc)
        series = centered.transpose(1, 2)
        return torch.cat((series, self.horizon_fill(series)), dim=-1), level

    def tokens(self, series: torch.Tensor) -> torch.Tensor:
        """Dual tokenization (Sec. A.3): ``[B, C, T] -> [B * C, T, d_model + value_dim]``."""
        batch, channels, length = series.shape
        mixed = series
        if self.training and self.channel_dropout:
            mixed = random_ratio_channel_dropout(series)
        channel = self.channel_token(mixed.transpose(1, 2)).unsqueeze(1).expand(-1, channels, -1, -1)
        value = series.unsqueeze(-1) * self.value_embedding[None, :, None, :]  # v_{c,t} = x_{c,t} V_c
        tokens = torch.cat((channel, value), dim=-1) + self.channel_embedding[None, :, None, :] + self.position
        return tokens.reshape(batch * channels, length, -1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        series, level = self.extend(x_enc)
        hidden = self.tokens(series)
        for block in self.blocks:
            hidden = block(hidden)
        horizon = hidden[:, -self.pred_len:, -self.value_dim:]
        horizon = horizon.reshape(x_enc.shape[0], self.enc_in, self.pred_len, self.value_dim)
        forecast = torch.einsum("bchd,cd->bhc", horizon, self.value_embedding)  # y_{c,t} = h_{c,t} V_c^T
        return restore_last_value(forecast, level)

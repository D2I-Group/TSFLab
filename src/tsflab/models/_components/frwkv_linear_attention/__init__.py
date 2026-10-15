"""FRWKV frequency-branch linear attention (RWKV-7-style delta-rule state scan).

Independent implementation of the linear-attention encoder of FRWKV
(arXiv 2512.07539, Sec. 2.4, Eqs. 4-9) as realized by the official code at
``yangqingyuan-byte/FRWKV@d6dbf171`` (``model/FRWKV.py``), which FRWKV+
(arXiv 2605.15690, Sec. 3.2) inherits unchanged. Nothing is copied.

Per head (size ``D``), over a token sequence ``z_1..z_T``:

    r, k, v   = MLP_r(z + mu_r), MLP_k(z + mu_k), MLP_v(z + mu_v)
    g         = sigmoid(LoRA_g(z + mu_g))                (output gate)
    a         = sigmoid(LoRA_a(z + mu_a))                (replacement strength)
    w         = exp(-exp(-0.3) * sigmoid(tanh(LoRA_w(z + mu_w))))   (decay)
    k_tilde   = normalize(k * kappa)                     (removal key, per head)
    k_hat     = k * lerp(1, a, m)                        (replacement key)
    G_t       = diag(w_t) - k_tilde_t (a_t * k_tilde_t)^T
    S_t       = G_t S_{t-1} + v_t k_hat_t^T,   y_t = S_t r_t
    o         = MLP_o(LayerNorm(y + 0.2 (r . k_hat) b * v) * g)

``mu`` are additive offsets (the code's "token shift" does not shift tokens).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer

_DECAY_RATE = math.exp(-0.3)
_BONUS_SCALE = 0.2
_EXPANSION = 1.2


def frwkv_state_scan(
    r: torch.Tensor,
    removal_key: torch.Tensor,
    replacement_key: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    strength: torch.Tensor,
) -> torch.Tensor:
    """Sequential state recursion of Eqs. 5-7 on ``[batch, tokens, heads, head_size]`` tensors.

    ``removal_key`` must already be unit-normalized per head. The state starts
    at zero; ``S_t = (diag(w_t) - k~_t (a_t * k~_t)^T) S_{t-1} + v_t k^_t^T`` and
    the output is ``y_t = S_t r_t``.
    """
    batch, tokens, heads, size = r.shape
    state = r.new_zeros(batch, heads, size, size)
    outputs = []
    for t in range(tokens):
        k_rm = removal_key[:, t]
        transition = torch.diag_embed(decay[:, t]) - k_rm.unsqueeze(-1) * (strength[:, t] * k_rm).unsqueeze(-2)
        state = transition @ state + v[:, t].unsqueeze(-1) * replacement_key[:, t].unsqueeze(-2)
        outputs.append((state @ r[:, t].unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, dim=1)


class _LowRankMLP(nn.Module):
    """``B(A(x)) (+ bias)`` with bias-free factors; the optional bias starts at zero."""

    def __init__(self, width: int, rank: int, bias: bool) -> None:
        super().__init__()
        self.A = nn.Linear(width, rank, bias=False)
        self.B = nn.Linear(rank, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.B(self.A(x))
        return out if self.bias is None else out + self.bias


def _projection(width: int) -> nn.Sequential:
    hidden = int(width * _EXPANSION)
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))


class FRWKVLinearAttention(nn.Module):
    """FRWKV linear-attention token mixer with the attention-layer call signature.

    ``forward(queries, keys=None, values=None, attn_mask=None, tau=None, delta=None)``
    mixes ``queries`` ``[batch, tokens, d_model]`` with itself (keys and values
    are ignored; there is no mask) and returns ``(output, None)``, so it plugs
    into ``transformer_encdec.EncoderLayer``. ``dropout`` is applied once to the
    output projection.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be a positive multiple of n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model, self.n_heads, self.head_size = d_model, n_heads, d_model // n_heads
        for name in ("r", "k", "v", "g", "a", "w"):
            self.register_parameter(f"mu_{name}", nn.Parameter(torch.randn(d_model) * 0.02))
        rank = max(64, d_model // 8)
        self.decay_lora = _LowRankMLP(d_model, rank, bias=True)
        self.strength_lora = _LowRankMLP(d_model, rank, bias=True)
        self.gate_lora = _LowRankMLP(d_model, rank, bias=False)
        self.receptance = _projection(d_model)
        self.key = _projection(d_model)
        self.value = _projection(d_model)
        self.output = _projection(d_model)
        self.removal_scale = nn.Parameter(torch.randn(d_model) * 0.05)
        self.strength_mix = nn.Parameter(torch.full((d_model,), 0.7))
        self.bonus = nn.Parameter(torch.full((d_model,), 1.2))
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def decay(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel decay ``exp(-e^-0.3 sigmoid(tanh(.)))``, inside about (0.58, 0.82)."""
        precursor = torch.tanh(self.decay_lora(x + self.mu_w))
        return torch.exp(-_DECAY_RATE * torch.sigmoid(precursor))

    def forward(self, queries, keys=None, values=None, attn_mask=None, tau=None, delta=None):
        x = queries
        batch, tokens, width = x.shape
        if width != self.d_model:
            raise ValueError(f"expected last width {self.d_model}, got {width}")
        heads = (batch, tokens, self.n_heads, self.head_size)
        r = self.receptance(x + self.mu_r)
        k = self.key(x + self.mu_k)
        v = self.value(x + self.mu_v)
        gate = torch.sigmoid(self.gate_lora(x + self.mu_g))
        strength = torch.sigmoid(self.strength_lora(x + self.mu_a))
        decay = self.decay(x)
        removal_key = F.normalize((k * self.removal_scale).view(heads), dim=-1)
        replacement_key = k * torch.lerp(torch.ones_like(strength), strength, self.strength_mix)
        r, v, replacement_key = r.view(heads), v.view(heads), replacement_key.view(heads)
        y = frwkv_state_scan(r, removal_key, replacement_key, v, decay.view(heads), strength.view(heads))
        bonus = (
            _BONUS_SCALE
            * (r * replacement_key).sum(dim=-1, keepdim=True)
            * v
            * self.bonus.view(self.n_heads, self.head_size)
        )
        out = self.norm((y + bonus).reshape(batch, tokens, width)) * gate
        return self.dropout(self.output(out)), None


class FRWKVSpectralBranch(nn.Module):
    """Residual frequency branch ``x + Linear_out(Encoder(Linear_in(x)))``.

    ``x`` is ``[batch, tokens, in_features]`` (FRWKV: one token per variable,
    ``in_features = embed_size * freq_bins``). The encoder stacks ``e_layers``
    ``transformer_encdec.EncoderLayer`` blocks around ``FRWKVLinearAttention``
    with a final LayerNorm; ``dropout`` is the encoder-layer rate and the
    attention output uses half of it.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        e_layers: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if in_features < 1 or e_layers < 1:
            raise ValueError("in_features and e_layers must be positive")
        self.proj_in = nn.Linear(in_features, d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FRWKVLinearAttention(d_model, n_heads, dropout * 0.5),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.proj_out = nn.Linear(d_model, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.proj_in(x))
        return x + self.proj_out(encoded)


__all__ = ["FRWKVLinearAttention", "FRWKVSpectralBranch", "frwkv_state_scan"]

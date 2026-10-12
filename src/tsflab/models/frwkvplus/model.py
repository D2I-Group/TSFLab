"""FRWKV+: periodic-aware adaptive gating on the FRWKV frequency backbone.

Independent implementation of arXiv 2605.15690 (Sec. 3, Eqs. 1-22) checked
against the official release ``yangqingyuan-byte/FRWKV-plus@08516fc0``
(``src/adaptive_phasegate_kbs/FRWKV/model/FRWKV_CrossBranchPhaseGateAdaptive.py``,
the paper's ``frwkv_crossbranchperiodicpositiongate_adaptive``); nothing is copied.

    X_emb                       RevIN + scalar-to-vector lift          (Eqs. 1-2)
    Z~_r, Z~_i                  FRWKV real/imag branches on rFFT        (Eqs. 3-5)
    C_r, C_i = mean_f Z~        branch summaries                        (Eq. 6)
    G0_{i->r}, G0_{r->i}        sigmoid MLP of the opposite summary     (Eq. 7)
    C_pos = PPCE(X_emb)         period-position router context          (Eqs. 8-12)
    Delta = tanh MLP([C_other, C_pos]),  T = sigmoid MLP([C_r, C_i, C_pos])  (Eqs. 13-15)
    G = 1 + G0 + clip(alpha, 0, 0.2) T Delta,  Z_bar = Z~ * G           (Eqs. 16-19)
    Y = RevIN^-1(Head(X_emb + irFFT(Z_bar)))                            (Eqs. 20-22)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.frwkv_linear_attention import FRWKVSpectralBranch
from tsflab.models._components.revin import RevIN

_ALPHA_MAX = 0.20


def _mlp(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.Linear(hidden, out_features))


class PeriodicPositionRouter(nn.Module):
    """PPCE router attention (Eqs. 9-12) on period-position tokens ``[B, N, P, E]`` -> ``[B, N, E]``."""

    def __init__(self, embed_size: int, num_routers: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.routers = nn.Parameter(torch.randn(num_routers, embed_size) * 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, variables, positions, width = tokens.shape
        x = self.norm(tokens.reshape(batch * variables, positions, width))
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scale = math.sqrt(width)
        router_weights = torch.softmax(self.routers @ k.transpose(1, 2) / scale, dim=-1)  # Eq. 10
        buffer = router_weights @ v  # [BN, R, E]
        token_weights = torch.softmax(q @ buffer.transpose(1, 2) / scale, dim=-1)  # Eq. 11
        context = (token_weights @ buffer).mean(dim=1)  # Eq. 12
        return self.out_proj(context).reshape(batch, variables, width)


class Model(nn.Module):
    """FRWKV+ forecaster for ``[batch, seq_len, enc_in]`` inputs.

    Backbone parameters and dropout fractions are those of FRWKV; the extra
    parameters are the period-position length ``period_len`` (P), the number of
    routers ``num_routers`` (R), the initial correction budget ``alpha_init`` and
    the initial trust bias ``trust_bias_init``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        d_ff: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        embed_size: int = 16,
        dropout: float = 0.2,
        activation: str = "gelu",
        period_len: int = 24,
        num_routers: int = 4,
        alpha_init: float = 0.10,
        trust_bias_init: float = -2.0,
        weighted_l1_loss: bool = True,
        loss_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if seq_len < 2 or pred_len < 1:
            raise ValueError("seq_len >= 2 and positive pred_len required")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.embed_size, self.period_len = embed_size, period_len
        # Training-objective settings read by spec.training_objective (official WeightedL1Loss).
        self.weighted_l1_loss, self.loss_alpha = bool(weighted_l1_loss), float(loss_alpha)
        self.freq_bins = seq_len // 2 + 1
        self.revin = RevIN(enc_in, affine=True)
        self.embedding = nn.Parameter(torch.randn(embed_size) * 0.1)
        branch = dict(
            in_features=self.freq_bins * embed_size,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout * 0.5,
            activation=activation,
        )
        self.real_branch = FRWKVSpectralBranch(**branch)
        self.imag_branch = FRWKVSpectralBranch(**branch)

        gate_hidden = max(32, embed_size * 8)
        trust_hidden = max(32, embed_size * 12)
        self.base_imag_to_real = _mlp(embed_size, gate_hidden, embed_size)
        self.base_real_to_imag = _mlp(embed_size, gate_hidden, embed_size)
        self.period_router = PeriodicPositionRouter(embed_size, num_routers)
        self.delta_imag_to_real = _mlp(2 * embed_size, gate_hidden, embed_size)
        self.delta_real_to_imag = _mlp(2 * embed_size, gate_hidden, embed_size)
        self.trust_real = _mlp(3 * embed_size, trust_hidden, embed_size)
        self.trust_imag = _mlp(3 * embed_size, trust_hidden, embed_size)
        # Start at the base-gate-only configuration: zero corrections, low trust.
        for head in (self.delta_imag_to_real, self.delta_real_to_imag):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        for head in (self.trust_real, self.trust_imag):
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, trust_bias_init)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.head = nn.Sequential(
            nn.Linear(seq_len * embed_size, d_ff),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(d_ff // 2, pred_len),
        )
        self.output_dropout = nn.Dropout(dropout * 0.5)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 2: ``[B, L, N] -> [B, N, L, E]`` (identity lift when E = 1)."""
        x = x.transpose(1, 2).unsqueeze(-1)
        return x if self.embed_size == 1 else x * self.embedding

    def period_position_tokens(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Eq. 8: average the embedded sequence over periods per within-period position.

        The length is extended to a multiple of P by appending the leading steps
        (cyclically, so P may exceed the input length).
        """
        batch, variables, length, width = x_emb.shape
        pad = (-length) % self.period_len
        if pad:
            index = torch.arange(pad, device=x_emb.device) % length
            x_emb = torch.cat([x_emb, x_emb.index_select(2, index)], dim=2)
        periods = x_emb.shape[2] // self.period_len
        return x_emb.reshape(batch, variables, periods, self.period_len, width).mean(dim=2)

    def encode_spectrum(self, part: torch.Tensor, branch: FRWKVSpectralBranch) -> torch.Tensor:
        """Eqs. 4-5: FRWKV residual branch over variable tokens of one part ``[B, N, E, F]``."""
        batch, variables, embed, bins = part.shape
        return branch(part.reshape(batch, variables, embed * bins)).reshape(batch, variables, embed, bins)

    def cross_branch_gates(
        self, real: torch.Tensor, imag: torch.Tensor, period_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. 6-7 and 13-17: final gates ``G_r``, ``G_i`` shaped ``[B, N, E]``."""
        real_ctx, imag_ctx = real.mean(dim=-1), imag.mean(dim=-1)
        base_real = torch.sigmoid(self.base_imag_to_real(imag_ctx))
        base_imag = torch.sigmoid(self.base_real_to_imag(real_ctx))
        delta_real = torch.tanh(self.delta_imag_to_real(torch.cat([imag_ctx, period_context], dim=-1)))
        delta_imag = torch.tanh(self.delta_real_to_imag(torch.cat([real_ctx, period_context], dim=-1)))
        joint = torch.cat([real_ctx, imag_ctx, period_context], dim=-1)
        trust_real = torch.sigmoid(self.trust_real(joint))
        trust_imag = torch.sigmoid(self.trust_imag(joint))
        alpha = self.alpha.clamp(0.0, _ALPHA_MAX)
        return 1.0 + base_real + alpha * trust_real * delta_real, 1.0 + base_imag + alpha * trust_imag * delta_imag

    def spectral_transform(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Eqs. 3-20: gated frequency-space transform back to ``[B, N, L, E]``."""
        period_context = self.period_router(self.period_position_tokens(x_emb))
        spectrum = torch.fft.rfft(x_emb.transpose(-1, -2), dim=-1, norm="ortho")
        real = self.encode_spectrum(spectrum.real, self.real_branch)
        imag = self.encode_spectrum(spectrum.imag, self.imag_branch)
        gate_real, gate_imag = self.cross_branch_gates(real, imag, period_context)
        gated = torch.complex(real * gate_real.unsqueeze(-1), imag * gate_imag.unsqueeze(-1))
        series = torch.fft.irfft(gated, n=self.seq_len, dim=-1, norm="ortho")
        return series.transpose(-1, -2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        x = self.revin(x_enc, "norm")
        x_emb = self.embed(x)
        hidden = x_emb + self.spectral_transform(x_emb)  # Eq. 21
        out = self.head(hidden.flatten(-2)).transpose(1, 2)
        return self.revin(self.output_dropout(out), "denorm")  # Eq. 22

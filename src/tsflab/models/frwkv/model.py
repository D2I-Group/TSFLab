"""FRWKV: frequency-domain linear attention for long-term forecasting.

Independent implementation of arXiv 2512.07539 (Sec. 2, Eqs. 1-9) following the
official code at ``yangqingyuan-byte/FRWKV@d6dbf171`` (``model/FRWKV.py``) where
the two differ; nothing is copied.

    X_norm = RevIN(X)                                  [B, L, N]
    X_emb  = X_norm (outer) a                           [B, N, L, E]   (scalar-to-vector lift)
    Z      = rFFT_L(X_emb^T, ortho) = Z_r + i Z_i       [B, N, E, F]   (Eq. 1)
    Z_r'   = Z_r + Branch_r(Z_r)  (tokens = variables, features = E*F)
    Z_i'   = Z_i + Branch_i(Z_i)                        (Sec. 2.4, Eqs. 4-9)
    X_fre  = irFFT_L(Z_r' + i Z_i', ortho)^T            (Eq. 2)
    Y      = RevIN^-1(MLP_head(flatten(X_emb + X_fre)))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.frwkv_linear_attention import FRWKVSpectralBranch
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """FRWKV forecaster for ``[batch, seq_len, enc_in]`` inputs.

    ``dropout`` is the configured rate; as in the official code the encoder
    layers use half of it, the attention output a quarter, and the head
    dropouts 0.3, 0.2 and 0.5 of it.
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
        weighted_l1_loss: bool = True,
        loss_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len >= 2 required")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.embed_size = embed_size
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
        """``[B, L, N] -> [B, N, L, E]``: each value times a learned vector (identity when E = 1)."""
        x = x.transpose(1, 2).unsqueeze(-1)
        return x if self.embed_size == 1 else x * self.embedding

    def encode_spectrum(self, part: torch.Tensor, branch: FRWKVSpectralBranch) -> torch.Tensor:
        """Residual branch over variable tokens of one spectral part ``[B, N, E, F]``."""
        batch, variables, embed, bins = part.shape
        return branch(part.reshape(batch, variables, embed * bins)).reshape(batch, variables, embed, bins)

    def spectral_transform(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Eqs. 1-2: rFFT over time, real/imag branches, irFFT back to ``[B, N, L, E]``."""
        spectrum = torch.fft.rfft(x_emb.transpose(-1, -2), dim=-1, norm="ortho")
        real = self.encode_spectrum(spectrum.real, self.real_branch)
        imag = self.encode_spectrum(spectrum.imag, self.imag_branch)
        series = torch.fft.irfft(torch.complex(real, imag), n=self.seq_len, dim=-1, norm="ortho")
        return series.transpose(-1, -2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        x = self.revin(x_enc, "norm")
        x_emb = self.embed(x)
        hidden = x_emb + self.spectral_transform(x_emb)
        out = self.head(hidden.flatten(-2)).transpose(1, 2)
        return self.revin(self.output_dropout(out), "denorm")

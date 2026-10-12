"""Client: cross-variable linear integrated enhanced Transformer (Gao et al., 2023).

Independent implementation from Section 3 and Algorithm 1 of the paper, with
omissions resolved against the official repository (daxin007/Client, revision
802c603f5ff3dd7814afd3e096f9082873373f1e, ``models/Client.py``).

Forecast, Eq. (2)-(4):
    H'      = RevIN(H)                                  (instance normalization)
    F_trans = Proj(Encoder(H'^T))^T                     (variate tokens of width L)
    F_lin   = Linear_L->O(H'^T)^T                       (channel-independent linear)
    F       = RevIN^-1(F_trans + w_lin * F_lin)         (w_lin learnable, per variate)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer


class Model(nn.Module):
    """Client forecaster for ``[batch, seq_len, enc_in]`` histories."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 32,
        dropout: float = 0.1,
        activation: str = "gelu",
        w_lin: float = 1.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths must be positive")
        if seq_len % n_heads:
            raise ValueError(
                "Client uses the lookback length as the token width, so seq_len must be "
                f"divisible by n_heads (got seq_len={seq_len}, n_heads={n_heads})"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        # No embedding and no positional encoding: the token width is seq_len (Sec. 3.1).
        d_model = seq_len
        self.revin = RevIN(enc_in, eps=1e-5, affine=True)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=dropout),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, pred_len)
        self.linear = nn.Linear(seq_len, pred_len)
        self.w_lin = nn.Parameter(torch.full((enc_in,), float(w_lin)))

    def cross_variable_transformer(self, normalized: torch.Tensor) -> torch.Tensor:
        """Eq. (1)-(2): attention across variate tokens, then project each to the horizon."""
        tokens, _ = self.encoder(normalized.transpose(1, 2))
        return self.projection(tokens).transpose(1, 2)

    def linear_module(self, normalized: torch.Tensor) -> torch.Tensor:
        """Eq. (3): one time-axis linear map shared by every variate."""
        return self.linear(normalized.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or tuple(x_enc.shape[1:]) != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in})")
        normalized = self.revin(x_enc, "norm")
        combined = self.cross_variable_transformer(normalized) + self.w_lin * self.linear_module(
            normalized
        )
        return self.revin(combined, "denorm")

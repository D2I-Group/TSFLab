"""Paper-driven local implementation of TimeMachine (four Mambas, two scales).

Pipeline (paper Sec. 3): RevIN, embedding E1 (L -> n1) and E2 (n1 -> n2), an outer
Mamba pair on the n1 embedding and an inner pair on the n2 embedding, projection
P1 (n2 -> n1), concatenation with the outer pair, and projection P2 (2*n1 -> T).
Under channel independence one Mamba of each pair treats the embedding axis as the
token axis (input width 1) and the other treats it as the width (one token).
Under channel mixing the tokens are the channels.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def _mamba(width: int, d_state: int, d_conv: int, expand: int) -> MambaBlock:
    """Kernel-free Mamba mixer of input width ``width`` (no norm, no residual)."""
    return MambaBlock(
        width,
        width * expand,
        math.ceil(width / 16),
        d_conv,
        d_state,
        reference_dt_init=True,
    )


class Model(nn.Module):
    """TimeMachine multivariate forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n1: int = 256,
        n2: int = 128,
        d_state: int = 256,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.05,
        revin: bool = True,
        ch_ind: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.ch_ind, self.residual = bool(ch_ind), bool(residual)
        # revin=False keeps the same statistics but drops the learnable affine
        self.norm = RevIN(enc_in, affine=bool(revin))
        self.embed1 = nn.Linear(seq_len, n1)
        self.embed2 = nn.Linear(n1, n2)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        # inner pair (n2 scale): mamba1 changes axis under channel independence
        self.mamba1 = _mamba(1 if ch_ind else n2, d_state, d_conv, expand)
        self.mamba2 = _mamba(n2, d_state, d_conv, expand)
        # outer pair (n1 scale)
        self.mamba3 = _mamba(n1, d_state, d_conv, expand)
        self.mamba4 = _mamba(1 if ch_ind else n1, d_state, d_conv, expand)
        self.proj1 = nn.Linear(n2, n1)
        self.proj2 = nn.Linear(2 * n1, pred_len)

    def _axis(self, x: torch.Tensor) -> torch.Tensor:
        """Swap token and width axes for the width-1 Mambas of channel independence."""
        return x.transpose(1, 2) if self.ch_ind else x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        x = self.norm(x_enc, "norm").transpose(1, 2)  # (B, C, L)
        if self.ch_ind:
            x = x.reshape(batch * self.enc_in, 1, self.seq_len)
        x = self.embed1(x)  # x(1): n1 embedding
        skip1 = x
        x = self.drop1(x)
        outer = self._axis(self.mamba4(self._axis(x))) + self.mamba3(x)
        x = self.embed2(x)  # x(2): n2 embedding
        skip2 = x
        x = self.drop2(x)
        inner = self._axis(self.mamba1(self._axis(x))) + self.mamba2(x)
        if self.residual:
            inner = inner + skip2
        x = self.proj1(inner)  # x(4)
        if self.residual:
            x = x + skip1
        x = self.proj2(torch.cat([x, outer], dim=-1))  # y = P2(x(5) || (x(4) + x(1)))
        if self.ch_ind:
            x = x.reshape(batch, self.enc_in, self.pred_len)
        return self.norm(x.transpose(1, 2), "denorm")

"""Temporal Parallel Gated Network (TPGN) built on the Parallel Gated Network (PGN).

Paper equations (Sections 3.1-3.2), per variate ``m`` with the series reshaped to
``R`` rows of ``P`` columns (``P`` the natural period, ``R * P = seq_len``):

    H   = HIE(Padding(X))                       historical information extraction
    G   = sigmoid(W_g [X, H] + b_g)
    H^  = tanh(W_t [X, H] + b_t)
    Out = G * H + (1 - G) * H^                  PGN, applied along the R axis   (1)

    X_long   = PGN(X_2D);   H_long  = Linear_long(X_long)   rows -> 1           (3)
    H_short  = Linear_row(X_2D);  H_glob = Linear_col(H_short)                   (4)
    Out      = Reshape(Linear([H_long, H_glob_repeated]))                        (5)

Every variate owns its parameters (the paper's per-variable formulation); inputs to
each linear map are the value channel concatenated with the calendar features.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.marks import adapt_tslib_marks
from tsflab.models._components.revin import RevIN

TIME_FEATURES = 4  # hourly Time-Series-Library continuous calendar features


class VariateLinear(nn.Module):
    """Independent affine map ``in_features -> out_features`` for every variate.

    Input ``[..., variates, in_features]``, output ``[..., variates, out_features]``.
    """

    def __init__(self, variates: int, in_features: int, out_features: int) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(variates, out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(variates, out_features).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...vi,voi->...vo", x, self.weight) + self.bias


class ParallelGatedNetwork(nn.Module):
    """PGN along the row axis of ``[batch, rows, cols, variates, features]``.

    Row ``r`` receives a linear summary of rows ``0..r-1`` (zero padded in front),
    which a single gate mixes with a candidate computed from the current row and
    that summary. Returns ``[batch, rows, cols, variates, hidden]``.
    """

    def __init__(self, rows: int, variates: int, features: int, hidden: int) -> None:
        super().__init__()
        if rows < 2:
            raise ValueError("PGN needs at least two rows to have a history")
        self.rows = rows
        self.window = rows - 1
        self.variates = variates
        self.features = features
        self.hidden = hidden
        bound = 1.0 / math.sqrt(features * self.window)
        self.hie_weight = nn.Parameter(
            torch.empty(variates, hidden, features, self.window).uniform_(-bound, bound)
        )
        self.hie_bias = nn.Parameter(torch.empty(variates, hidden).uniform_(-bound, bound))
        self.gate = VariateLinear(variates, features + hidden, 2 * hidden)

    def history(self, x: torch.Tensor) -> torch.Tensor:
        """HIE: ``[B, R, P, V, F] -> [B, R, P, V, hidden]`` from strictly earlier rows."""
        batch, _, cols, variates, features = x.shape
        zeros = x.new_zeros(batch, self.window, cols, variates, features)
        padded = torch.cat([zeros, x], dim=1)[:, :-1]  # [B, window + R - 1, P, V, F]
        windows = padded.unfold(1, self.window, 1)  # [B, R, P, V, F, window]
        return torch.einsum("brpvfj,vofj->brpvo", windows, self.hie_weight) + self.hie_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hist = self.history(x)
        gates = self.gate(torch.cat([x, hist], dim=-1))
        update, candidate = gates.chunk(2, dim=-1)
        update = torch.sigmoid(update)
        return hist * update + (1.0 - update) * torch.tanh(candidate)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period: int = 24,
        d_model: int = 64,
        norm: bool = True,
        use_short_branch: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("sizes must be positive")
        if seq_len % period or pred_len % period:
            raise ValueError("seq_len and pred_len must be multiples of period")
        if seq_len // period < 2:
            raise ValueError("seq_len must cover at least two periods")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.period, self.d_model = period, d_model
        self.rows, self.pred_rows = seq_len // period, pred_len // period
        self.use_short_branch = use_short_branch
        features = 1 + TIME_FEATURES

        self.revin = RevIN(enc_in, eps=1e-5, affine=False, enabled=norm)
        self.pgn = ParallelGatedNetwork(self.rows, enc_in, features, d_model)
        self.long_pool = VariateLinear(enc_in, d_model * self.rows, d_model)  # Linear_long
        if use_short_branch:
            self.short_row = VariateLinear(enc_in, features * period, d_model)  # Linear_short^row
            self.short_col = VariateLinear(enc_in, d_model * self.rows, d_model)  # Linear_short^col
        branches = 2 if use_short_branch else 1
        self.head = VariateLinear(enc_in, branches * d_model, self.pred_rows)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        batch = x_enc.shape[0]
        marks = adapt_tslib_marks(x_mark_enc, embed_type="timeF", freq="h")
        if marks is None:
            marks = x_enc.new_zeros(batch, self.seq_len, TIME_FEATURES)
        if marks.shape[:2] != (batch, self.seq_len):
            raise ValueError("x_mark_enc must cover the same history as x_enc")
        values = self.revin(x_enc, "norm")
        # X_2D: [B, R, P, V, 1 + time features]; calendar features repeat across variates.
        values = values.reshape(batch, self.rows, self.period, self.enc_in, 1)
        calendar = marks.to(values.dtype).reshape(batch, self.rows, self.period, 1, TIME_FEATURES)
        grid = torch.cat([values, calendar.expand(-1, -1, -1, self.enc_in, -1)], dim=-1)

        long_rows = self.pgn(grid)  # [B, R, P, V, d]
        long_branch = self.long_pool(  # aggregate the R rows of every column -> [B, P, V, d]
            long_rows.permute(0, 2, 3, 4, 1).reshape(batch, self.period, self.enc_in, -1)
        )
        if self.use_short_branch:
            rows = grid.permute(0, 1, 3, 4, 2).reshape(batch, self.rows, self.enc_in, -1)
            short_rows = self.short_row(rows)  # [B, R, V, d]
            pooled = self.short_col(short_rows.permute(0, 2, 3, 1).reshape(batch, self.enc_in, -1))
            short_branch = pooled.unsqueeze(1).expand(-1, self.period, -1, -1)  # [B, P, V, d]
            long_branch = torch.cat([short_branch, long_branch], dim=-1)
        out = self.head(long_branch)  # [B, P, V, R_f]
        forecast = out.permute(0, 3, 1, 2).reshape(batch, self.pred_len, self.enc_in)
        return self.revin(forecast, "denorm")

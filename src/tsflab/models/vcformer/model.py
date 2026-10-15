"""VCformer: variable correlation attention with lagged cross-correlation plus a
Koopman temporal detector over inverted variate tokens.

Independent implementation from Section 3 (Eqs. 3-11, Algorithm 1) of Yang, Zhu
and Chen, "VCformer: Variable Correlation Transformer with Inherent Lagged
Correlation for Multivariate Time Series Forecasting" (arXiv 2405.11470, IJCAI
2024), after reading the pinned official code (``CSyyn/VCformer`` at
``67e8dc8c``, no license file) to resolve omissions; nothing is copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.marks import encoder_timef_marks, tslib_time_feature_dimension
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def lagged_cross_correlation(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Eqs. (3)-(4) evaluated with the FFT of Eq. (11).

    ``queries`` and ``keys`` are ``[B, N, T]`` (one row per variate token, ``T``
    the axis the lags run over). Returns ``[B, N, N, T]`` with
    ``R[b, i, j, tau] = sum_t q_i[(t + tau) mod T] * k_j[t]`` (circular lag),
    computed as ``irfft(F(q_i) * conj(F(k_j)))``.
    """
    length = queries.shape[-1]
    q_spec = torch.fft.rfft(queries, dim=-1).unsqueeze(2)  # [B, N, 1, F]
    k_spec = torch.fft.rfft(keys, dim=-1).unsqueeze(1)  # [B, 1, N, F]
    return torch.fft.irfft(q_spec * torch.conj(k_spec), n=length, dim=-1)


class VariableCorrelationAttention(nn.Module):
    """VCA (Section 3.3): lagged cross-correlation map aggregated over all lags.

    ``COR(q_i, k_j) = sum_tau lambda_tau R_{q_i, k_j}(tau)`` (Eq. 5) with learnable
    ``lambda`` (initialised to ``1 / d_model``, which reproduces the official
    mean over lags at initialisation), then ``softmax(COR / sqrt(d_model)) V``
    (Eq. 6 with the official scale). One head over the whole width.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.lag_weights = nn.Parameter(torch.full((d_model,), 1.0 / d_model))
        self.scale = 1.0 / math.sqrt(d_model)

    def correlation_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pre-softmax ``[B, N, N]`` scores ``COR(Q, K)`` of Eq. (5)."""
        lagged = lagged_cross_correlation(self.query(tokens), self.key(tokens))
        return (lagged * self.lag_weights).sum(dim=-1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.correlation_map(tokens) * self.scale, dim=-1)
        attended = torch.einsum("bij,bjd->bid", weights, self.value(tokens))
        return self.out(attended), weights


class KoopmanMLP(nn.Module):
    """Encoder / decoder MLP of the KTD: Linear, tanh, dropout, Linear."""

    def __init__(self, f_in: int, f_out: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, f_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_koopman_operator(embeddings: torch.Tensor) -> torch.Tensor:
    """Eq. (8) in row-vector form: ``K = pinv(Z_back) Z_fore`` per sequence.

    ``embeddings`` is ``[batch, P, M]`` (``P`` snapshots). ``Z_back`` holds
    snapshots ``1..P-1`` and ``Z_fore`` snapshots ``2..P``, so ``z_{j+1} ~ z_j K``.
    A non-finite operator is replaced by the identity (official safeguard).
    """
    operator = torch.linalg.pinv(embeddings[:, :-1]) @ embeddings[:, 1:]
    bad = ~torch.isfinite(operator).flatten(1).all(dim=1)
    if bad.any():
        eye = torch.eye(operator.shape[-1], dtype=operator.dtype, device=operator.device)
        operator = torch.where(bad[:, None, None], eye, operator)
    return operator


def koopman_rollout(last: torch.Tensor, operator: torch.Tensor, steps: int) -> torch.Tensor:
    """Eq. (9): ``z_{P+t} = z_P K^t`` for ``t = 1..steps``; ``last`` is ``[batch, M]``."""
    state = last.unsqueeze(1)
    predictions = []
    for _ in range(steps):
        state = state @ operator
        predictions.append(state)
    return torch.cat(predictions, dim=1)


class KoopmanTemporalDetector(nn.Module):
    """KTD (Section 3.4): segment each token into ``d_model / S`` snapshots of length
    ``S`` (Eq. 7), encode them into a Koopman space, fit the operator by eDMD
    (Eq. 8), roll it forward ``d_model / S`` steps (Eq. 9) and decode (Eq. 10)."""

    def __init__(self, d_model: int, snap_size: int, proj_dim: int, hidden_dim: int, mlp_dropout: float) -> None:
        super().__init__()
        self.snap_size = snap_size
        self.num_snapshots = d_model // snap_size
        self.encoder = KoopmanMLP(snap_size, proj_dim, hidden_dim, mlp_dropout)
        self.decoder = KoopmanMLP(proj_dim, snap_size, hidden_dim, mlp_dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_tokens, width = tokens.shape
        snapshots = tokens.reshape(batch * n_tokens, self.num_snapshots, self.snap_size)
        embedded = self.encoder(snapshots)  # [B*N, P, M]
        operator = fit_koopman_operator(embedded)
        future = koopman_rollout(embedded[:, -1], operator, self.num_snapshots)
        return self.decoder(future).reshape(batch, n_tokens, width)


class VCformerLayer(nn.Module):
    """Algorithm 1 lines 6 and 8: post-norm residual VCA, then post-norm residual KTD."""

    def __init__(
        self, d_model: int, snap_size: int, proj_dim: int, hidden_dim: int, dropout: float, mlp_dropout: float
    ) -> None:
        super().__init__()
        self.attention = VariableCorrelationAttention(d_model)
        self.ktd = KoopmanTemporalDetector(d_model, snap_size, proj_dim, hidden_dim, mlp_dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ktd_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(tokens)
        tokens = self.attention_norm(tokens + self.dropout(attended))
        tokens = self.ktd_norm(tokens + self.dropout(self.ktd(tokens)))
        return tokens, weights


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        e_layers: int = 3,
        snap_size: int = 16,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        koopman_dropout: float = 0.05,
        use_norm: bool = True,
        use_marks: bool = True,
        freq: str = "h",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_marks = use_marks
        self.freq = freq
        if use_marks:
            tslib_time_feature_dimension(freq)  # validates the calendar layout early
        self.normalizer = RevIN(enc_in, eps=1e-5, affine=False, enabled=use_norm)
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            VCformerLayer(d_model, snap_size, proj_dim, hidden_dim, dropout, koopman_dropout) for _ in range(e_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, pred_len)

    def calendar_tokens(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        return encoder_timef_marks(x_mark_enc, seq_len=self.seq_len, freq=self.freq, enabled=self.use_marks)

    def encode(self, x: torch.Tensor, x_mark_enc: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized window to encoded variate (and calendar) tokens ``[B, N(+marks), d_model]``."""
        tokens = self.embedding(x, self.calendar_tokens(x_mark_enc))
        for layer in self.layers:
            tokens, _ = layer(tokens)
        return self.final_norm(tokens)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        x = self.normalizer(x_enc, "norm")
        forecast = self.projection(self.encode(x, x_mark_enc)).transpose(1, 2)[:, :, : self.enc_in]
        return self.normalizer(forecast, "denorm")

"""Mamba-ProbTSF: a dual-network Gaussian forecaster (S-Mamba mean, MLP scale).

Independent implementation from Section 2.2 (Eqs. 1, 6-10) of Pessoa et al.,
"Mamba time series forecasting with uncertainty quantification" (arXiv
2503.10873, Machine Learning: Science and Technology 6, 035012, 2025), after
reading the pinned official code (``PessoaP/Mamba-ProbTSF`` at ``a69d7c3c``, no
license file) to resolve omissions; nothing is copied. The selective SSM is the
cataloged pure-PyTorch ``mamba`` block instead of the ``mamba_ssm`` kernels.

``forward`` returns ``[batch, pred_len, channels, 2]`` holding the Gaussian
location ``mu`` (first network) and standard deviation ``sigma > 0`` (second
network), trained jointly with the Gaussian negative log-likelihood of Eq. (8).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.marks import encoder_timef_marks, tslib_time_feature_dimension
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window

#: Floor added to the softplus standard deviation (official value).
SIGMA_FLOOR = 1e-8


def _scan(d_model: int, d_state: int, d_conv: int, expand: int) -> MambaBlock:
    return MambaBlock(
        d_model=d_model,
        d_inner=expand * d_model,
        dt_rank=math.ceil(d_model / 16),
        d_conv=d_conv,
        d_state=d_state,
        reference_dt_init=True,
    )


class SMambaEncoderLayer(nn.Module):
    """S-Mamba block over variate tokens: forward plus flipped Mamba scan added
    residually, LayerNorm, pointwise feed-forward network, LayerNorm."""

    def __init__(
        self, d_model: int, d_state: int, d_ff: int, d_conv: int, expand: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.forward_scan = _scan(d_model, d_state, d_conv, expand)
        self.backward_scan = _scan(d_model, d_state, d_conv, expand)
        self.scan_norm = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scanned = self.forward_scan(tokens) + self.backward_scan(tokens.flip(1)).flip(1)
        tokens = self.scan_norm(tokens + scanned)
        hidden = self.dropout(self.activation(self.ff_in(tokens)))
        return self.ffn_norm(tokens + self.dropout(self.ff_out(hidden)))


class SMambaForecaster(nn.Module):
    """Mean network ``mu_{1:T}(x_{1:P})`` (Section 2.2): S-Mamba with inverted
    variate (and calendar) tokens inside non-affine RevIN."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        d_state: int,
        d_ff: int,
        e_layers: int,
        d_conv: int,
        expand: int,
        dropout: float,
        activation: str,
        use_norm: bool,
    ) -> None:
        super().__init__()
        self.enc_in = enc_in
        self.normalizer = RevIN(enc_in, eps=1e-5, affine=False, enabled=use_norm)
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            SMambaEncoderLayer(d_model, d_state, d_ff, d_conv, expand, dropout, activation)
            for _ in range(e_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x_enc: torch.Tensor, calendar: torch.Tensor | None) -> torch.Tensor:
        x = self.normalizer(x_enc, "norm")
        tokens = self.embedding(x, calendar)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)
        forecast = self.projection(tokens).transpose(1, 2)[:, :, : self.enc_in]
        return self.normalizer(forecast, "denorm")


class SigmaMLP(nn.Module):
    """Uncertainty network ``sigma_{1:T}(x_{1:P})`` (Section 2.2): a fully connected
    network applied to each raw variate history, with a softplus last layer:
    ``softplus(W3 GELU(W2 GELU(W1 x))) + 1e-8``."""

    def __init__(self, seq_len: int, pred_len: int, hidden: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(seq_len, hidden)
        self.fc_hidden = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, pred_len)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc_hidden(F.gelu(self.fc_in(x_enc.transpose(1, 2)))))
        return (F.softplus(self.fc_out(h)) + SIGMA_FLOOR).transpose(1, 2)


def gaussian_negative_log_likelihood(target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Elementwise Eq. (8) term ``((x - mu) / sigma)^2 / 2 + log sigma`` (no constant)."""
    return 0.5 * ((target - mu) / sigma) ** 2 + torch.log(sigma)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        d_state: int = 16,
        d_ff: int = 512,
        e_layers: int = 3,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
        sigma_network: str = "mlp",
        sigma_hidden: int = 512,
        use_marks: bool = True,
        freq: str = "h",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.sigma_network = sigma_network
        self.use_marks = use_marks
        self.freq = freq
        # Probabilistic output contract: (loc, scale) per horizon step and channel.
        self.output_type = "distribution"
        self.distribution_family = "gaussian"
        if use_marks:
            tslib_time_feature_dimension(freq)  # validates the calendar layout early
        backbone = dict(
            seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, d_model=d_model, d_state=d_state,
            d_ff=d_ff, e_layers=e_layers, d_conv=d_conv, expand=expand, dropout=dropout,
            activation=activation, use_norm=use_norm,
        )
        self.mean_network = SMambaForecaster(**backbone)
        self.sigma_net: nn.Module = (
            SigmaMLP(seq_len, pred_len, sigma_hidden) if sigma_network == "mlp" else SMambaForecaster(**backbone)
        )

    def calendar_tokens(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        return encoder_timef_marks(x_mark_enc, seq_len=self.seq_len, freq=self.freq, enabled=self.use_marks)

    def sigma(self, x_enc: torch.Tensor, calendar: torch.Tensor | None) -> torch.Tensor:
        """Forecast standard deviation ``[B, pred_len, C]`` (strictly positive)."""
        if self.sigma_network == "mlp":
            return self.sigma_net(x_enc)
        return F.softplus(self.sigma_net(x_enc, calendar)) + SIGMA_FLOOR

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        calendar = self.calendar_tokens(x_mark_enc)
        mu = self.mean_network(x_enc, calendar)
        return torch.stack((mu, self.sigma(x_enc, calendar)), dim=-1)

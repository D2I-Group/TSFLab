"""MoGU: Mixture-of-Gaussians with Uncertainty-based Gating (arXiv 2510.07459).

An ensemble of ``k`` probabilistic iTransformer experts. Each expert predicts a
mean and a per-step, per-variable variance (Eq. 14); the experts are combined by
inverse-variance (precision) weights (Eq. 10) instead of a learned gate, trained
with the precision-weighted sum of per-expert Gaussian NLLs (Eq. 11), and the
model reports the mixture mean (Eq. 7) and the law-of-total-variance spread
(Eq. 12) as a Gaussian ``(loc, scale)`` forecast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer


def precision_weights(variance: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. (10): ``w_i = sigma_i^-2 / sum_j sigma_j^-2`` over the expert axis (dim 1)."""
    precision = 1.0 / (variance + eps)
    return precision / precision.sum(dim=1, keepdim=True)


def mixture_moments(
    means: torch.Tensor, variances: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eqs. (7) and (12) over the expert axis (dim 1).

    Returns the mixture mean, the aleatoric term ``sum_i w_i sigma_i^2`` and the
    epistemic term ``sum_i w_i (y_hat - y_i)^2``.
    """
    mean = (weights * means).sum(dim=1)
    aleatoric = (weights * variances).sum(dim=1)
    epistemic = (weights * (mean.unsqueeze(1) - means).pow(2)).sum(dim=1)
    return mean, aleatoric, epistemic


def gaussian_nll(
    mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, eps: float
) -> torch.Tensor:
    """Eq. (6): ``0.5 * (log max(sigma^2, eps) + (mu - y)^2 / max(sigma^2, eps))``, elementwise."""
    variance = variance.clamp_min(eps)
    return 0.5 * (torch.log(variance) + (mean - target).pow(2) / variance)


def mogu_loss(
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Eq. (11): ``mean( sum_i w_i * L_NLLG(y; y_i, sigma_i^2) )``.

    ``means``, ``variances`` and ``weights`` are ``[B, k, H, C]`` and ``target`` is
    ``[B, H, C]``; the gate is not detached, so the NLL also trains the variances
    through the weights.
    """
    nll = gaussian_nll(means, variances, target.unsqueeze(1), eps)
    return (weights * nll).sum(dim=1).mean()


class UncertaintyHead(nn.Module):
    """Eq. (14): ``sigma^2 = softplus(f'(g(x)))`` per variate token and horizon step.

    ``f'`` is an MLP with one hidden layer of width ``d_model`` (``"mlp"``) or a
    single linear map (``"linear"``, the Table 8 ablation).
    """

    def __init__(self, d_model: int, pred_len: int, head_type: str) -> None:
        super().__init__()
        if head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, pred_len)
            )
        else:
            self.net = nn.Linear(d_model, pred_len)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [B, C, d_model] -> variance [B, H, C]
        return F.softplus(self.net(tokens), threshold=20.0).transpose(1, 2)


class GaussianITransformerExpert(nn.Module):
    """One probabilistic expert: iTransformer mean path plus the uncertainty head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        activation: str,
        head_type: str,
    ) -> None:
        super().__init__()
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
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
        self.uncertainty_head = UncertaintyHead(d_model, pred_len, head_type)

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. (13) x -> f(g(x)): g is the inverted encoder, f the linear projection.
        tokens, _ = self.encoder(self.embedding(normalized, None), attn_mask=None)
        mean = self.projection(tokens).transpose(1, 2)
        return mean, self.uncertainty_head(tokens)


class Model(nn.Module):
    """MoGU over ``num_experts`` iTransformer experts; returns ``[B, H, C, 2]`` (loc, scale)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_experts: int = 3,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        head_type: str = "mlp",
        use_norm: bool = True,
        denorm_variance: bool = False,
        nll_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.use_norm = use_norm
        self.denorm_variance = denorm_variance
        self.nll_eps = nll_eps
        self.output_type = "distribution"
        self.distribution_family = "gaussian"
        self.experts = nn.ModuleList(
            GaussianITransformerExpert(
                seq_len, pred_len, d_model, n_heads, e_layers, d_ff, dropout, activation, head_type
            )
            for _ in range(num_experts)
        )

    def expert_outputs(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-expert means and variances, each ``[B, k, H, C]`` (Eq. 15)."""
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in})")
        if self.use_norm:
            # Instance normalization of every variate over the look-back window.
            normalized, mean, stdev = standardize(x_enc)
        else:
            normalized = x_enc
        means, variances = [], []
        for expert in self.experts:
            expert_mean, expert_variance = expert(normalized)
            if self.use_norm:
                expert_mean = destandardize(expert_mean, mean, stdev)
                if self.denorm_variance:
                    expert_variance = expert_variance * stdev.pow(2)
            means.append(expert_mean)
            variances.append(expert_variance)
        return torch.stack(means, dim=1), torch.stack(variances, dim=1)

    def combine(self, means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
        """Precision gate (Eq. 10), mixture mean (Eq. 7) and total variance (Eq. 12)."""
        weights = precision_weights(variances)
        loc, aleatoric, epistemic = mixture_moments(means, variances, weights)
        return torch.stack([loc, torch.sqrt(aleatoric + epistemic)], dim=-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        means, variances = self.expert_outputs(x_enc)
        return self.combine(means, variances)

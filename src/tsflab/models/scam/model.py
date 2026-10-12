"""Local SCAM: Self-Correction with Adaptive Mask on an MLP forecaster.

Paper map (Yang et al., NeurIPS 2025, arXiv 2502.14704): the predictor
``f(x; theta)`` (here an ensemble of the paper's two-layer MLP, optionally with
Spectral Norm Regularization, Section 3.4, Eq. 6) is trained jointly with a
reconstruction network ``g(y; phi)`` (conv-concat layers plus a point-wise FFN,
Section 2.2 and Appendix D) whose outputs act as self-supervised pseudo labels
under the adaptive-mask loss (Section 3.3, Eqs. 4-5). ``g`` is used only in
training; ``forward`` returns the mean of the predictor ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_FORMS = ("official", "paper")


def _l2_normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector / (vector.norm() + eps)


class SpectralNormLinear(nn.Linear):
    """Eq. (6) without the scale: ``y = (W / sigma_max(W)) x + b``.

    ``sigma_max`` is estimated by one power iteration per call on the detached
    weight with a persistent left singular vector ``u``, so gradients flow only
    through the numerator (pinned official ``SNLinear``).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)
        self.register_buffer("u", torch.randn(1, out_features))

    def normalized_weight(self) -> torch.Tensor:
        weight = self.weight.detach()
        with torch.no_grad():
            v = _l2_normalize(self.u @ weight)
            u = _l2_normalize(v @ weight.T)
            sigma = ((u @ weight) * v).sum()
            self.u.copy_(u)
        return self.weight / sigma

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.normalized_weight(), self.bias)


class MLPPredictor(nn.Module):
    """Paper backbone: per-window standardized two-layer SiLU MLP over time.

    ``[B, C, L] -> [B, C, H]``; with ``snr`` the first layer is a spectrally
    normalized ``L -> d/2`` map followed by ``d/2 -> d``.
    """

    def __init__(self, seq_len: int, pred_len: int, hidden: int, snr: bool, instance_norm: bool) -> None:
        super().__init__()
        self.instance_norm = instance_norm
        if snr:
            layers: list[nn.Module] = [SpectralNormLinear(seq_len, hidden // 2), nn.Linear(hidden // 2, hidden)]
        else:
            layers = [nn.Linear(seq_len, hidden)]
        self.net = nn.Sequential(*layers, nn.SiLU(), nn.Linear(hidden, pred_len))

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        if not self.instance_norm:
            return self.net(series)
        mean = series.mean(dim=-1, keepdim=True)
        scale = series.std(dim=-1, keepdim=True) + 1e-6
        return self.net((series - mean) / scale) * scale + mean


class ConvConcatEmbedding(nn.Module):
    """Conv-concat layers of ``g`` (Appendix D.1), ``[B, C, T] -> [B, C, T, d]``.

    The series (right zero-padded to a multiple of ``2^layers``) is repeated to
    ``a`` channels; each layer applies InstanceNorm, a stride-2 kernel-3 conv
    doubling the channels, and SiLU. Every layer output ``[a * 2^k, T / 2^k]``
    is transposed and unfolded back to ``[T, a]`` (position ``l * 2^k + c``),
    concatenated with the input over layers, cropped to ``T``, and projected.
    """

    def __init__(self, layers: int, width: int, multiplier: int) -> None:
        super().__init__()
        self.layers, self.multiplier = layers, multiplier
        channels = multiplier
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.norms.append(nn.InstanceNorm1d(channels))
            self.convs.append(nn.Conv1d(channels, 2 * channels, kernel_size=3, stride=2, padding=1))
            channels *= 2
        self.projection = nn.Linear((layers + 1) * multiplier, width)

    def unfolded_features(self, series: torch.Tensor) -> torch.Tensor:
        """Concatenated multi-resolution features ``[B, C, T, a * (layers + 1)]``."""
        batch, channels, length = series.shape
        block = 2**self.layers
        signal = F.pad(series, (0, (block - length % block) % block))
        padded = signal.shape[-1]
        hidden = signal.reshape(batch * channels, 1, padded).expand(-1, self.multiplier, -1)
        features = [hidden.transpose(1, 2)]
        for norm, conv in zip(self.norms, self.convs):
            hidden = F.silu(conv(norm(hidden)))
            per_group = hidden.shape[1] // self.multiplier
            unfolded = hidden.reshape(batch * channels, self.multiplier, per_group, -1)
            features.append(unfolded.permute(0, 3, 2, 1).reshape(batch * channels, padded, self.multiplier))
        stacked = torch.stack(features, dim=-1)  # [B*C, T_pad, a, layers + 1]
        stacked = stacked.reshape(batch, channels, padded, -1)
        return stacked[:, :, :length]

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        return self.projection(self.unfolded_features(series))


class CandidateDecoder(nn.Module):
    """Point-wise FFN output of ``g`` (Appendix D.2): N relabel candidates.

    ``SiLU`` then the ``d`` features are split into ``N`` groups of ``d / N``;
    group ``n`` is mapped to one value per step by its own ``Linear(d/N, 1)``.
    All heads start from the same initial weights (pinned official code) and
    are trained independently. ``[B, C, T, d] -> [B, C, N, T]``.
    """

    def __init__(self, width: int, candidates: int) -> None:
        super().__init__()
        self.candidates, self.group = candidates, width // candidates
        template = nn.Linear(self.group, 1)
        self.weight = nn.Parameter(template.weight.detach().reshape(1, self.group).repeat(candidates, 1))
        self.bias = nn.Parameter(template.bias.detach().repeat(candidates))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        groups = F.silu(features).unflatten(-1, (self.candidates, self.group))
        decoded = (groups * self.weight).sum(dim=-1) + self.bias  # [B, C, T, N]
        return decoded.permute(0, 1, 3, 2)


def scam_loss(y: torch.Tensor, y_rec: torch.Tensor, y_hat: torch.Tensor, form: str = "official") -> torch.Tensor:
    """Adaptive-mask co-training loss over ``[B, C, N, H]`` candidates; ``y`` is ``[B, C, H]``.

    With ``A = y_rec - y_hat``, ``B = y_rec - y``, ``M = (A B > 0)`` (the pseudo
    label lies outside the segment between ``y`` and ``y_hat``) and
    ``M< = |A| < |B|`` (both without gradient), Eq. (5) is
    ``|y - y_hat| (1 - M) + 2 |A| M< M + 2 |B| (1 - M<) M``: where ``M`` holds, the
    supervised term is replaced by the auxiliary ``2 min(|A|, |B|)`` of Eq. (4);
    elsewhere only the supervised term remains. ``official`` (pinned code) uses
    the squared error for the supervised term and adds ``A^2 M``.
    """
    if form not in LOSS_FORMS:
        raise ValueError(f"loss form must be one of {LOSS_FORMS}, got {form!r}")
    target = y.unsqueeze(2)
    pred_gap = (y_hat - y_rec).abs()
    rec_gap = (target - y_rec).abs()
    with torch.no_grad():
        outside = (y_rec - target) * (y_rec - y_hat) > 0
        closer = pred_gap < rec_gap
    outside_f, closer_f = outside.to(y_hat.dtype), closer.to(y_hat.dtype)
    if form == "paper":
        supervised = (y_hat - target).abs() * (1 - outside_f)
        auxiliary = 2 * pred_gap * closer_f * outside_f + 2 * rec_gap * (1 - closer_f) * outside_f
        return (supervised + auxiliary).mean()
    loss_pred = 2 * outside_f * closer_f * pred_gap + outside_f * (y_hat - y_rec).pow(2)
    loss_rec = 2 * outside_f * (1 - closer_f) * rec_gap
    loss_tar = (1 - outside_f) * (y_hat - target).pow(2)
    return loss_pred.mean() + loss_rec.mean() + loss_tar.mean()


class Model(nn.Module):
    """SCAM: ensemble of MLP predictors plus a training-only relabeling network."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        candidates: int = 8,
        hidden: int = 128,
        snr: bool = False,
        instance_norm: bool = True,
        rec_layers: int = 4,
        rec_width: int = 128,
        rec_multiplier: int = 4,
        loss_form: str = "official",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths, channels, and widths must be positive")
        if instance_norm and seq_len < 2:
            raise ValueError("per-window standardization needs seq_len >= 2")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.loss_form = loss_form
        self.predictors = nn.ModuleList(
            MLPPredictor(seq_len, pred_len, hidden, snr, instance_norm) for _ in range(candidates)
        )
        self.embedding = ConvConcatEmbedding(rec_layers, rec_width, rec_multiplier)
        self.decoder = CandidateDecoder(rec_width, candidates)

    def candidate_forecasts(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``f(x; theta_n)`` for every ensemble member: ``[B, L, C] -> [B, C, N, H]``."""
        series = x_enc.transpose(1, 2)
        return torch.stack([predictor(series) for predictor in self.predictors], dim=2)

    def relabel(self, target: torch.Tensor) -> torch.Tensor:
        """Pseudo labels ``g(y; phi)``: ``[B, H, C] -> [B, C, N, H]``."""
        return self.decoder(self.embedding(target.transpose(1, 2)))

    def training_loss(self, x_enc: torch.Tensor, target: torch.Tensor, channels: slice) -> tuple[torch.Tensor, torch.Tensor]:
        """Co-training step: (ensemble-mean forecast ``[B, H, C]``, SCAM loss).

        ``target`` holds the supervised channels selected by ``channels``.
        """
        y_hat = self.candidate_forecasts(x_enc)
        y_rec = self.relabel(target)
        loss = scam_loss(target.transpose(1, 2), y_rec, y_hat[:, channels], self.loss_form)
        return y_hat.mean(dim=2).transpose(1, 2), loss

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must be [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        return self.candidate_forecasts(x_enc).mean(dim=2).transpose(1, 2)

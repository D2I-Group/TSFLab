"""MoSSL: multi-modality spatio-temporal forecasting via self-supervised learning.

Independent implementation of Deng et al., "Multi-Modality Spatio-Temporal
Forecasting via Self-Supervised Learning" (IJCAI 2024), checked against the
official repository at ``2ca992ea`` (no license; nothing copied).

Tensor layout: TSFLab supplies ``x_enc [B, T, N]`` with ``N = M * L``
(``M`` modalities, ``L`` nodes).  Channels are modality-major: channel
``m * L + l`` holds modality ``m`` at node ``l``.  Internally the tensor is
``[B, C, M, L, T]`` (hidden channels, modalities, nodes, time).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def receptive_length(layers: int, kernel_size: int) -> int:
    """History length consumed by ``layers`` valid dilated convs (dilation ``2**i``)."""
    return 1 + (kernel_size - 1) * (2 ** layers - 1)


def _pointwise(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(channels, channels, 1), nn.ReLU())


class AxisAttention(nn.Module):
    """Spatial (Eq. 3, ``axis=3``) or modality (Eq. 2, ``axis=2``) attention.

    Query, key and value are 1x1 convs with ReLU (``f_1, f_2, f_3``); scores are
    scaled by ``sqrt(d_z)`` and normalized over the chosen axis independently for
    every other position; a 1x1 conv with ReLU follows, as in the official code.
    """

    def __init__(self, channels: int, axis: int) -> None:
        super().__init__()
        if axis not in (2, 3):
            raise ValueError("axis must be 2 (modality) or 3 (node)")
        self.axis = axis
        self.channels = channels
        self.query = _pointwise(channels)
        self.key = _pointwise(channels)
        self.value = _pointwise(channels)
        self.out = _pointwise(channels)

    def _tokens(self, h: torch.Tensor) -> torch.Tensor:
        """``[B, C, M, L, T]`` -> ``[B, other, other, T, axis, C]`` with the axis second to last."""
        return h.movedim(1, -1).movedim(self.axis - 1, -2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q, k, v = (self._tokens(f(h)) for f in (self.query, self.key, self.value))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.channels)
        y = torch.softmax(scores, dim=-1) @ v
        return self.out(y.movedim(-2, self.axis - 1).movedim(-1, 1))


class MoSTLayer(nn.Module):
    """One MoST Encoder layer: ``[H, MA(H), SA(H)]`` then the gated temporal conv (Eq. 4).

    Filter and gate convs span all modalities and ``kernel_size`` steps at
    ``dilation`` (valid, no padding) and emit ``M * C`` channels folded back to
    ``[C, M]``; residual and skip 1x1 convs follow.
    """

    def __init__(self, channels: int, num_modalities: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.spatial = AxisAttention(channels, axis=3)
        self.modality = AxisAttention(channels, axis=2)
        conv = dict(
            in_channels=3 * channels,
            out_channels=num_modalities * channels,
            kernel_size=(num_modalities, 1, kernel_size),
            dilation=(1, 1, dilation),
        )
        self.filter_conv = nn.Conv3d(**conv)
        self.gate_conv = nn.Conv3d(**conv)
        self.residual_conv = nn.Conv3d(channels, channels, 1)
        self.skip_conv = nn.Conv3d(channels, channels, 1)

    def fold(self, y: torch.Tensor) -> torch.Tensor:
        """``[B, M*C, 1, L, T'] -> [B, C, M, L, T']`` (output channel ``c * M + m``)."""
        b, _, _, n, t = y.shape
        return y.reshape(b, -1, self.num_modalities, n, t)

    def forward(self, h: torch.Tensor):
        h_hat = torch.cat([self.spatial(h), self.modality(h), h], dim=1)
        g = self.fold(torch.tanh(self.filter_conv(h_hat))) * self.fold(torch.sigmoid(self.gate_conv(h_hat)))
        return self.residual_conv(g), self.skip_conv(g)


class MoSTEncoder(nn.Module):
    """Stacked MoST layers with residual connections and summed skip outputs."""

    def __init__(self, channels: int, num_modalities: int, layers: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            MoSTLayer(channels, num_modalities, kernel_size, 2 ** i) for i in range(layers)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        skip = None
        for layer in self.layers:
            out, sk = layer(h)
            h = out + h[..., -out.shape[-1]:]
            skip = sk if skip is None else sk + skip[..., -sk.shape[-1]:]
        return skip


class MultiModalityAugmentation(nn.Module):
    """Multi-modality Data Augmentation (Sec. 3.2).

    Modality relevance ``phi`` is the softmax over modalities of ``C`` learned
    1x1 queries of the original representation, averaged over the queries
    (official code; Eq. 5 uses one vector ``w_0``).  A fixed budget of
    ``mask_ratio * B * L * M`` (node, modality) cells is drawn with replacement
    with probability proportional to ``1 - phi`` and zeroed in the projected
    input ``H_in`` (detached, as in the code); the MoST embedding
    ``e_t + e_n + e_m`` is concatenated and projected with a 1x1 conv and ReLU.
    """

    def __init__(self, channels: int, seq_len: int, num_nodes: int, num_modalities: int, mask_ratio: float) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.relevance = nn.Conv2d(channels, channels, 1)
        self.temporal_embedding = nn.Parameter(torch.empty(channels, seq_len))
        self.spatial_embedding = nn.Parameter(torch.empty(channels, num_nodes))
        self.modality_embedding = nn.Parameter(torch.empty(channels, num_modalities))
        for p in (self.temporal_embedding, self.spatial_embedding, self.modality_embedding):
            nn.init.xavier_normal_(p)
        self.proj = nn.Sequential(nn.Conv3d(2 * channels, channels, 1), nn.ReLU())

    def modality_relevance(self, rep: torch.Tensor) -> torch.Tensor:
        """``rep [B, C, M, L, T']`` -> ``phi [B, L*T', M]`` (rows sum to one)."""
        b, c, m = rep.shape[:3]
        grid = rep.permute(0, 1, 3, 4, 2).reshape(b, c, -1, m)
        return torch.softmax(self.relevance(grid), dim=-1).mean(dim=1)

    def mask(self, h_in: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Zero sampled ``(b, node, modality)`` cells of ``h_in [B, C, M, L, T]``."""
        b, _, m, n, _ = h_in.shape
        weights = (1.0 - phi).reshape(-1)
        count = int(b * n * m * self.mask_ratio)
        keep = torch.ones(b * n * m, device=h_in.device, dtype=h_in.dtype)
        if count > 0 and float(weights.sum()) > 0:
            picked = torch.multinomial(weights, count, replacement=True)
            keep[picked] = 0.0
        keep = keep.reshape(b, n, m).permute(0, 2, 1)  # [B, M, L]
        return h_in * keep[:, None, :, :, None]

    def embedding(self) -> torch.Tensor:
        """MoST embedding ``E [1, C, M, L, T]``."""
        c = self.temporal_embedding.shape[0]
        return (
            self.temporal_embedding.reshape(1, c, 1, 1, -1)
            + self.spatial_embedding.reshape(1, c, 1, -1, 1)
            + self.modality_embedding.reshape(1, c, -1, 1, 1)
        )

    def forward(self, h_in: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
        phi = self.modality_relevance(rep).detach()
        masked = self.mask(h_in.detach(), phi)
        e = self.embedding().expand_as(masked)
        return self.proj(torch.cat([masked, e], dim=1))


class GlobalSSL(nn.Module):
    """Global Self-Supervised Learning (Sec. 3.3, Eqs. 6-9).

    From the augmented representation: memberships ``gamma`` (softmax of a
    bias-free linear map of ``vec(H~)``, Eq. 7) and per-cluster, per-channel
    means and scales (linear maps of each channel's slice, Eq. 8, ``exp`` read as
    the scale as in the code).  The loss is the negative log-likelihood of the
    original representation under that mixture with diagonal Gaussians (Eq. 9),
    averaged over positions and the batch.  Both representations are
    L2-normalized over channels first, as in the code.
    """

    def __init__(self, channels: int, positions: int, num_components: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(positions * channels, num_components, bias=False)
        nn.init.xavier_uniform_(self.gamma.weight)
        self.mu = nn.Linear(positions, num_components)
        self.log_sigma = nn.Linear(positions, num_components)

    def mixture(self, rep_aug: torch.Tensor):
        """``[B, C, P]`` -> ``gamma [B, K]``, ``mu, sigma [B, C, K]``."""
        gamma = torch.softmax(self.gamma(rep_aug.reshape(rep_aug.shape[0], -1)), dim=-1)
        return gamma, self.mu(rep_aug), torch.exp(self.log_sigma(rep_aug))

    @staticmethod
    def log_likelihood(h: torch.Tensor, gamma, mu, sigma) -> torch.Tensor:
        """``log sum_k gamma_k N(h_p | mu_k, diag sigma_k^2)`` per position -> ``[B, P]``."""
        z = (h.unsqueeze(-1) - mu.unsqueeze(2)) / sigma.unsqueeze(2)  # [B, C, P, K]
        log_comp = (-torch.log(sigma).unsqueeze(2) - 0.5 * LOG_2PI - 0.5 * z.pow(2)).sum(dim=1)
        return torch.logsumexp(log_comp + torch.log(gamma).unsqueeze(1), dim=-1)

    def forward(self, rep: torch.Tensor, rep_aug: torch.Tensor) -> torch.Tensor:
        b, c = rep.shape[:2]
        h = F.normalize(rep, dim=1).reshape(b, c, -1)
        h_aug = F.normalize(rep_aug, dim=1).reshape(b, c, -1)
        gamma, mu, sigma = self.mixture(h_aug)
        return -self.log_likelihood(h, gamma, mu, sigma).mean()


class ModalitySSL(nn.Module):
    """Modality Self-Supervised Learning (Sec. 3.4, Eqs. 10-11).

    ``r = H * W_1 + H~ * W_2`` (weights per node-modality cell and channel, as in
    the code), ``c_m = sigmoid(mean_nodes r)``; positives pair ``r_{n,m}`` with
    ``c_m``, negatives pair ``r_{n,m'}`` (``m' != m``, a random non-zero cyclic
    modality shift) with ``c_m``; scores are a shared bilinear form and the loss
    is binary cross-entropy with logits.
    """

    def __init__(self, channels: int, num_nodes: int, num_modalities: int) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.w1 = nn.Parameter(torch.empty(num_modalities * num_nodes, channels))
        self.w2 = nn.Parameter(torch.empty(num_modalities * num_nodes, channels))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        self.score = nn.Bilinear(channels, channels, 1)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def fuse(self, rep: torch.Tensor, rep_aug: torch.Tensor):
        """``[B, C, M, L, 1]`` pair -> ``r [B, M, L, C]`` and ``c_m [B, M, C]``."""
        b, c, m, n, _ = rep.shape

        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 2, 3, 4, 1).reshape(b, m * n, c)

        r = (flat(rep) * self.w1 + flat(rep_aug) * self.w2).reshape(b, m, n, c)
        return r, torch.sigmoid(r.mean(dim=2))

    def loss(self, r: torch.Tensor, cm: torch.Tensor, shift: int) -> torch.Tensor:
        negatives = torch.roll(r, shifts=shift, dims=1)
        cm = cm.unsqueeze(2).expand_as(r).contiguous()
        positive = self.score(r.contiguous(), cm)
        negative = self.score(negatives.contiguous(), cm)
        logits = torch.cat([positive, negative], dim=-1)
        labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=-1)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def forward(self, rep: torch.Tensor, rep_aug: torch.Tensor) -> torch.Tensor:
        if self.num_modalities < 2:
            return rep.new_zeros(())
        r, cm = self.fuse(rep, rep_aug)
        shift = int(torch.randint(1, self.num_modalities, (1,)))
        return self.loss(r, cm, shift)


class Model(nn.Module):
    """MoSSL forecaster with the four-input TSFLab interface.

    ``forward`` returns ``[B, pred_len, N]`` from the original view only.  In
    training mode (with ``self_supervised``) the augmented view is encoded by the
    shared encoder and ``aux_loss = L_g + L_c`` is stored for the trainer, which
    adds it to the regression loss (Eq. 13).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_modalities: int = 1,
        hidden_channels: int = 48,
        num_components: int = 4,
        layers: int = 4,
        kernel_size: int = 2,
        mask_ratio: float = 0.1,
        self_supervised: bool = True,
    ) -> None:
        super().__init__()
        if seq_len != receptive_length(layers, kernel_size):
            raise ValueError(
                "MoSSL consumes exactly 1 + (kernel_size - 1) * (2**layers - 1) = "
                f"{receptive_length(layers, kernel_size)} steps; got seq_len={seq_len}"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_modalities = num_modalities
        self.num_nodes = enc_in // num_modalities
        self.self_supervised = self_supervised
        c, m, n = hidden_channels, num_modalities, self.num_nodes

        self.input_proj = nn.Sequential(
            nn.Conv3d(1, c // 2, 1), nn.ReLU(), nn.Conv3d(c // 2, c, 1), nn.ReLU()
        )
        self.encoder = MoSTEncoder(c, m, layers, kernel_size)
        self.predictor = nn.Sequential(
            nn.ReLU(), nn.Conv3d(c, c, 1), nn.ReLU(), nn.Conv3d(c, pred_len, 1)
        )
        self.augmentation = MultiModalityAugmentation(c, seq_len, n, m, mask_ratio)
        self.gssl = GlobalSSL(c, m * n, num_components)
        self.mssl = ModalitySSL(c, n, m)
        self.aux_loss: torch.Tensor | None = None

    def to_tensor(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``[B, T, M*L]`` (modality-major) -> ``[B, 1, M, L, T]``."""
        b, t, _ = x_enc.shape
        return x_enc.reshape(b, t, self.num_modalities, self.num_nodes).permute(0, 2, 3, 1).unsqueeze(1)

    def self_supervised_loss(self, h_in: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
        """``L_g + L_c`` from the augmented view through the shared encoder."""
        rep_aug = self.encoder(self.augmentation(h_in, rep))
        return self.gssl(rep, rep_aug) + self.mssl(rep, rep_aug)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(f"expected x_enc [B, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        h_in = self.input_proj(self.to_tensor(x_enc))
        rep = self.encoder(h_in)  # [B, C, M, L, 1]
        y = self.predictor(rep)  # [B, pred_len, M, L, 1]
        if self.training and self.self_supervised:
            self.aux_loss = self.self_supervised_loss(h_in, rep)
        else:
            self.aux_loss = None
        return y.reshape(x_enc.shape[0], self.pred_len, self.enc_in)

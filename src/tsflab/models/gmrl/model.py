"""GMRL: Gaussian Mixture Representation Learning for tensor time series.

Independent implementation of Deng et al., "Learning Gaussian Mixture
Representations for Tensor Time Series Forecasting" (IJCAI 2023), checked against
the official repository at ``ef37b8a4`` (no license; nothing copied).

Tensor layout: TSFLab supplies ``x_enc [B, T, N]`` with ``N = S * L``
(``S`` sources, ``L`` locations).  Channels are source-major: channel
``s * L + l`` holds source ``s`` at location ``l``.  Internally the tensor is
``[B, C, S, L, T]`` (hidden channels, sources, locations, time).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def receptive_length(layers: int, kernel_size: int) -> int:
    """History length consumed by ``layers`` valid dilated convs (dilation ``2**i``)."""
    return 1 + (kernel_size - 1) * (2**layers - 1)


class GMRE(nn.Module):
    """Gaussian Mixture Representation Extractor (Sec. 3.2, Eqs. 2-7).

    For every hidden channel ``i`` the whole ``S x L x T`` slice ``vec(H[..., i])``
    is mapped to ``K`` prior weights ``alpha`` (softmax, Eq. 3), means ``mu`` and
    scales ``sigma = exp(.)`` (Eq. 4); the projections are shared across channels
    as in the official code.  Every scalar is assigned to its MAP cluster (Eq. 5)
    and normalized with that cluster's ``(mu, sigma)`` (Cluster Norm).
    """

    def __init__(self, num_components: int, flat_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_components = num_components
        self.eps = eps
        self.alpha = nn.Linear(flat_size, num_components)
        self.mu = nn.Linear(flat_size, num_components)
        self.log_sigma = nn.Linear(flat_size, num_components)

    def mixture_parameters(self, h: torch.Tensor):
        """``h [B, C, P]`` -> ``alpha, mu, sigma``, each ``[B, C, K]``."""
        alpha = torch.softmax(self.alpha(h), dim=-1)
        return alpha, self.mu(h), torch.exp(self.log_sigma(h))

    @staticmethod
    def component_log_pdf(h: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """``log N(h | mu_k, sigma_k^2)`` for ``h [B, C, P]`` -> ``[B, C, P, K]``."""
        z = (h.unsqueeze(-1) - mu.unsqueeze(2)) / sigma.unsqueeze(2)
        return -torch.log(sigma).unsqueeze(2) - 0.5 * LOG_2PI - 0.5 * z.pow(2)

    @staticmethod
    def cluster_loss(alpha, log_component, log_posterior) -> torch.Tensor:
        """Eq. 7: ``KL(Q(z) || P(z)) - E_data E_{P(z|h)} log P(h|z)``.

        ``Q(z)`` is the posterior averaged over the ``S x L x T`` positions of a
        channel; both terms are averaged over channels and the batch.
        """
        posterior = log_posterior.exp()
        q = posterior.mean(dim=2)
        kl = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(alpha.clamp_min(1e-12)))).sum(-1)
        expected_log_lik = (posterior * log_component).sum(-1).mean(dim=2)
        return (kl - expected_log_lik).mean()

    def forward(self, x: torch.Tensor):
        b, c = x.shape[:2]
        h = x.reshape(b, c, -1)
        alpha, mu, sigma = self.mixture_parameters(h)
        log_component = self.component_log_pdf(h, mu, sigma)
        log_joint = log_component + torch.log(alpha).unsqueeze(2)
        log_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)
        labels = log_posterior.argmax(dim=-1)
        # Cluster Norm: (H - mu_k) / (sigma_k + eps) with k the MAP cluster of each scalar.
        mu_k = torch.gather(mu, 2, labels)
        sigma_k = torch.gather(sigma, 2, labels)
        normed = (h - mu_k) / (sigma_k + self.eps)
        loss = self.cluster_loss(alpha, log_component, log_posterior)
        return normed.reshape_as(x), loss


class GMRETELayer(nn.Module):
    """One GMRE-TE layer: ``H_gm = [GMRE(H), H]``, then the gated Temporal Encoder (Eq. 8).

    The filter and gate convolutions span all sources and ``kernel_size`` steps at
    ``dilation`` (valid, no padding) and emit ``S * C`` channels folded back to
    ``[C, S]``, as in the official code; ``f_conv`` is the 1x1 residual conv.
    """

    def __init__(
        self,
        channels: int,
        num_sources: int,
        num_locations: int,
        time_len: int,
        num_components: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.num_sources = num_sources
        self.gmre = GMRE(num_components, num_sources * num_locations * time_len)
        conv = dict(
            in_channels=2 * channels,
            out_channels=num_sources * channels,
            kernel_size=(num_sources, 1, kernel_size),
            dilation=(1, 1, dilation),
        )
        self.filter_conv = nn.Conv3d(**conv)
        self.gate_conv = nn.Conv3d(**conv)
        self.residual_conv = nn.Conv3d(channels, channels, 1)
        self.skip_conv = nn.Conv3d(channels, channels, 1)

    def fold(self, y: torch.Tensor) -> torch.Tensor:
        """``[B, S*C, 1, L, T'] -> [B, C, S, L, T']`` (output channel ``c * S + s``)."""
        b, _, _, n, t = y.shape
        return y.reshape(b, -1, self.num_sources, n, t)

    def forward(self, x: torch.Tensor):
        normed, loss = self.gmre(x)
        h_gm = torch.cat([normed, x], dim=1)
        h = self.fold(torch.tanh(self.filter_conv(h_gm))) * self.fold(torch.sigmoid(self.gate_conv(h_gm)))
        return self.residual_conv(h), self.skip_conv(h), loss


class HRA(nn.Module):
    """Hidden Representation Augmenter (Sec. 3.4, Eq. 9) and Predictor (Eq. 10).

    The flattened skip representation queries a learned memory bank; the
    attention-weighted prototype is projected back to the representation's
    shape and concatenated before the two-layer ReLU predictor.
    """

    def __init__(self, channels: int, flat_size: int, memory_size: int, pred_len: int, use_memory: bool) -> None:
        super().__init__()
        self.use_memory = use_memory
        if use_memory:
            self.memory = nn.Parameter(torch.empty(memory_size, channels))
            self.query = nn.Linear(flat_size, channels)
            self.value = nn.Linear(channels, flat_size)
        self.out1 = nn.Conv3d(2 * channels if use_memory else channels, channels, 1)
        self.out2 = nn.Conv3d(channels, pred_len, 1)

    def augment(self, h_sc: torch.Tensor) -> torch.Tensor:
        q = self.query(h_sc.reshape(h_sc.shape[0], -1))
        phi = torch.softmax(q @ self.memory.t(), dim=-1)
        h_me = self.value(phi @ self.memory).reshape_as(h_sc)
        return torch.cat([h_sc, h_me], dim=1)

    def forward(self, h_sc: torch.Tensor) -> torch.Tensor:
        h = self.augment(h_sc) if self.use_memory else h_sc
        return self.out2(F.relu(self.out1(F.relu(h))))


class Model(nn.Module):
    """GMRL forecaster with the four-input TSFLab interface.

    ``forward`` returns ``[B, pred_len, N]``. In training mode the cluster
    objective ``cluster_weight * mean_layers L_cluster`` is stored on
    ``aux_loss``, which the trainer adds to the regression loss (Eq. 12).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_sources: int = 1,
        hidden_channels: int = 24,
        num_components: int = 17,
        layers: int = 4,
        kernel_size: int = 2,
        memory_size: int = 8,
        use_memory: bool = True,
        cluster_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if seq_len != receptive_length(layers, kernel_size):
            raise ValueError(
                "GMRL consumes exactly 1 + (kernel_size - 1) * (2**layers - 1) = "
                f"{receptive_length(layers, kernel_size)} steps; got seq_len={seq_len}"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_sources = num_sources
        self.num_locations = enc_in // num_sources
        self.cluster_weight = cluster_weight
        width = 2 * hidden_channels
        s, n = num_sources, self.num_locations

        # Tensor Time Series Embedding (Sec. 3.1): e_t + e_l + e_s next to f(X).
        self.input_proj = nn.Conv3d(1, hidden_channels, 1)
        self.temporal_embedding = nn.Parameter(torch.empty(hidden_channels, seq_len))
        self.location_embedding = nn.Parameter(torch.empty(hidden_channels, n))
        self.source_embedding = nn.Parameter(torch.empty(hidden_channels, s))
        self.embed_proj = nn.Conv3d(width, width, 1)

        self.layers = nn.ModuleList()
        time_len = seq_len
        for i in range(layers):
            dilation = 2**i
            self.layers.append(GMRETELayer(width, s, n, time_len, num_components, kernel_size, dilation))
            time_len -= dilation * (kernel_size - 1)
        self.hra = HRA(width, width * s * n, memory_size, pred_len, use_memory)
        self.aux_loss: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Official training-script initialization: Xavier-uniform for every
        multi-dimensional parameter, ``U(0, 1)`` for every vector."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def to_tensor(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``[B, T, S*L]`` (source-major) -> ``[B, 1, S, L, T]``."""
        b, t, _ = x_enc.shape
        return x_enc.reshape(b, t, self.num_sources, self.num_locations).permute(0, 2, 3, 1).unsqueeze(1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """``H = relu(conv([f(X), E]))`` with ``E = e_t + e_l + e_s``."""
        h = self.input_proj(x)
        c = h.shape[1]
        ttse = (
            self.temporal_embedding.reshape(1, c, 1, 1, -1)
            + self.location_embedding.reshape(1, c, 1, -1, 1)
            + self.source_embedding.reshape(1, c, -1, 1, 1)
        )
        return F.relu(self.embed_proj(torch.cat([h, ttse.expand_as(h)], dim=1)))

    def encode(self, h: torch.Tensor):
        """Stacked GMRE-TE layers with residual and summed skip connections."""
        skip = None
        losses = []
        for layer in self.layers:
            out, sk, loss = layer(h)
            h = out + h[..., -out.shape[-1] :]
            skip = sk if skip is None else sk + skip[..., -sk.shape[-1] :]
            losses.append(loss)
        return skip, torch.stack(losses).mean()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(f"expected x_enc [B, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        skip, cluster_loss = self.encode(self.embed(self.to_tensor(x_enc)))
        y = self.hra(skip)  # [B, pred_len, S, L, 1]
        self.aux_loss = self.cluster_weight * cluster_loss if self.training else None
        return y.reshape(x_enc.shape[0], self.pred_len, self.enc_in)

"""VolDy-VAE: volatility-dynamics VAE for probabilistic forecasting (Wang et al., 2026).

RevIN-normalized history patches are encoded into a diagonal Gaussian latent per
patch (Eq. 3); the flattened past latents are projected directly to future patch
latents (Eq. 4). A shared location-scale decoder maps every latent patch to a
de-normalized mean (Eq. 5) and to a positive scale produced by a GRU volatility
state (Eqs. 6-7) whose final past state initializes the future scale path.
Training minimizes the heteroscedastic Gaussian NLL of the reconstruction and the
forecast plus a beta-weighted KL term (Eqs. 8-9); the forecast is the set of
empirical quantiles of latent-and-observation samples.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.empirical_quantiles import empirical_quantiles
from tsflab.models._components.revin import RevIN


def patch_mlp(f_in: int, f_out: int, hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    """ReLU MLP with ``max(depth, 2)`` linear layers and dropout after each hidden layer.

    This is the depth convention of the K2VAE MLP the official code imports:
    ``depth`` counts every linear layer, with a minimum of two.
    """
    layers: list[nn.Module] = [nn.Linear(f_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, f_out))
    return nn.Sequential(*layers)


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    sigma_min: float = 1e-6,
    sigma_max: float = 1e3,
) -> torch.Tensor:
    """Eq. (9) ``1/(TC) sum_{t,c} [log sigma + (u - mu)^2 / (2 sigma^2)]``, batch-averaged.

    ``sigma`` is clamped to ``[sigma_min, sigma_max]`` first, as in the official code.
    """
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    nll = sigma.log() + (target - mu).square() / (2.0 * sigma.square())
    return nll.mean(dim=(1, 2)).mean()


def latent_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``KL(N(mu, exp(logvar)) || N(0, I))`` summed over the latent axis, averaged elsewhere."""
    return (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1)).mean()


class VolatilityDynamics(nn.Module):
    """Eqs. (6)-(7): ``h_t = GRU(h_{t-1}, Z_t)`` and ``sigma_t = Softplus(Linear(h_t)) + xi``."""

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, xi: float) -> None:
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.softplus = nn.Softplus()
        self.xi = xi

    def forward(self, z: torch.Tensor, h0: torch.Tensor | None = None):
        """``z [B, T, D]`` (one step per patch) -> ``sigma [B, T, out_dim]``, ``h_T [1, B, H]``."""
        states, last = self.gru(z, h0)
        return self.softplus(self.proj(states)) + self.xi, last


class Model(nn.Module):
    """``forward`` returns empirical quantiles ``[B, pred_len, enc_in, Q]`` of ``num_samples`` draws."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        quantile_levels: list[float],
        patch_len: int = 24,
        dynamic_dim: int = 128,
        hidden_layers: int = 3,
        hidden_dim: int = 256,
        vol_hidden_dim: int = 128,
        dropout: float = 0.05,
        beta: float = 0.01,
        num_samples: int = 100,
        xi: float = 1e-6,
    ) -> None:
        super().__init__()
        levels = list(quantile_levels)
        if not levels or any(not 0.0 < q < 1.0 for q in levels) or levels != sorted(set(levels)):
            raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
        self.output_type = "quantile"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.dynamic_dim = dynamic_dim
        self.beta = beta
        self.num_samples = num_samples
        self.register_buffer("quantile_levels", torch.tensor(levels), persistent=False)

        self.past_patches = math.ceil(seq_len / patch_len)  # N
        self.future_patches = math.ceil(pred_len / patch_len)  # M
        self.padding_len = self.past_patches * patch_len - seq_len
        patch_width = patch_len * enc_in

        self.revin = RevIN(enc_in)
        # Encoder E_phi (Eq. 3): one patch [P*C] -> (mu_z, log sigma_z^2), each [D].
        self.encoder = patch_mlp(patch_width, 2 * dynamic_dim, hidden_dim, hidden_layers, dropout)
        # Latent dynamics (Eq. 4): flatten N past latents, project to M future latents.
        flat = self.past_patches * dynamic_dim
        self.latent_projection = patch_mlp(flat, self.future_patches * dynamic_dim, flat // 2, 2, dropout)
        # Shared dual-head decoder: location MLP (Eq. 5) and GRU scale path (Eqs. 6-7).
        self.location_head = patch_mlp(dynamic_dim, patch_width, hidden_dim, hidden_layers, dropout)
        self.scale_head = VolatilityDynamics(dynamic_dim, vol_hidden_dim, patch_width, xi)

    # ----------------------------------------------------------------- encoder
    def patchify(self, normalized: torch.Tensor) -> torch.Tensor:
        """``[B, L, C] -> [B, N, P*C]``; a short history is front-filled with its own tail."""
        batch = normalized.shape[0]
        if self.padding_len > 0:
            normalized = torch.cat([normalized[:, -self.padding_len :], normalized], dim=1)
        patches = normalized.reshape(batch, self.past_patches, self.patch_len, self.enc_in)
        return patches.reshape(batch, self.past_patches, -1)

    def encode(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RevIN, patching and E_phi: ``[B, L, C] -> (mu_z, logvar_z)``, each ``[B, N, D]``."""
        mu, logvar = self.encoder(self.patchify(self.revin(x_enc, "norm"))).chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # ----------------------------------------------------------------- decoder
    def _unpatch(self, values: torch.Tensor, length: int) -> torch.Tensor:
        """``[B, T, P*C] -> [B, T*P, C]`` truncated to the first ``length`` steps."""
        batch, steps = values.shape[0], values.shape[1] * self.patch_len
        return values.reshape(batch, steps, self.enc_in)[:, :length]

    def decode_normalized(self, z: torch.Tensor):
        """Past latents ``[B, N, D]`` -> normalized means and scales of past and future.

        Returns ``(x_loc, y_loc, x_sigma, y_sigma)``. The location head is shared
        by both horizons (Eq. 5, before RevIN inversion); the scale path runs the
        GRU over the past patches and continues from its final state over the
        projected future patches (Eqs. 6-7).
        """
        batch = z.shape[0]
        x_loc = self._unpatch(self.location_head(z), self.seq_len)
        x_sigma_patches, h_vol = self.scale_head(z)
        x_sigma = self._unpatch(x_sigma_patches, self.seq_len)
        z_future = self.latent_projection(z.reshape(batch, -1))
        z_future = z_future.reshape(batch, self.future_patches, self.dynamic_dim)
        y_loc = self._unpatch(self.location_head(z_future), self.pred_len)
        y_sigma = self._unpatch(self.scale_head(z_future, h_vol)[0], self.pred_len)
        return x_loc, y_loc, x_sigma, y_sigma

    def decode(self, z: torch.Tensor):
        """``(x_mu, y_mu, x_sigma, y_sigma)``: means de-normalized by RevIN (Eq. 5).

        Scales keep the output of Eq. (7) without RevIN inversion, as in the paper
        and the official code.
        """
        x_loc, y_loc, x_sigma, y_sigma = self.decode_normalized(z)
        return self.revin(x_loc, "denorm"), self.revin(y_loc, "denorm"), x_sigma, y_sigma

    # ---------------------------------------------------------------- sampling
    def sample(self, x_enc: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """Draw ``Z ~ q(Z|P)`` then ``Y ~ N(y_mu, y_sigma^2)``: ``[B, K, pred_len, C]``."""
        count = num_samples or self.num_samples
        batch = x_enc.shape[0]
        mu, logvar = self.encode(x_enc)
        z = self.reparameterize(mu.repeat_interleave(count, 0), logvar.repeat_interleave(count, 0))
        _, y_loc, _, y_sigma = self.decode_normalized(z)  # [B*K, H, C], window-major
        # Every draw of a window shares that window's RevIN statistics.
        y_loc = y_loc.reshape(batch, count * self.pred_len, self.enc_in)
        y_mu = self.revin(y_loc, "denorm").reshape(batch, count, self.pred_len, self.enc_in)
        y_sigma = y_sigma.reshape(batch, count, self.pred_len, self.enc_in)
        return y_mu + torch.randn_like(y_mu) * y_sigma

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return empirical_quantiles(self.sample(x_enc), self.quantile_levels, dim=1)

    # -------------------------------------------------------------- objective
    def training_loss(self, x_enc: torch.Tensor, target: torch.Tensor, target_start: int = 0) -> torch.Tensor:
        """Eq. (8): ``NLL(X) + NLL(Y) + beta * KL`` with one reparameterized latent draw.

        ``target`` is the forecast target ``[B, pred_len, C']`` holding the model
        channels from ``target_start`` on (``-1`` for an ``MS`` target).
        """
        mu, logvar = self.encode(x_enc)
        x_mu, y_mu, x_sigma, y_sigma = self.decode(self.reparameterize(mu, logvar))
        rec = gaussian_nll(x_enc, x_mu, x_sigma)
        pred = gaussian_nll(target, y_mu[..., target_start:], y_sigma[..., target_start:])
        return rec + pred + self.beta * latent_kl(mu, logvar)

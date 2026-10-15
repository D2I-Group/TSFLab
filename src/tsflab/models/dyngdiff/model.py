"""DynG-Diff: state-aware dynamic guidance for an unconditional diffusion backbone.

Three stages (paper Sec. 4, Fig. 1):

1. ``pretrain``: an unconditional DDPM noise predictor ``eps_theta`` (S4 residual
   network, App. B.1) is fit to whole windows ``[context; horizon]`` with the simple
   noise MSE (Eq. 2) and then frozen.
2. ``policy_loss`` (the training objective): a small Conv1d State-Aware Policy
   Network reads ``[x_t ; x0_hat]`` and the step ``t`` and regresses the clipped
   z-scored log-precision of the frozen backbone's one-step estimate (Eqs. 8-10).
3. ``forward``: guided DDPM reverse sampling. At every step the policy output is
   mapped to the guidance matrix ``A_t = exp(Z_t) / mean(exp(Z_t))`` (Eq. 7), used
   detached as the local precision of an asymmetric-Laplace observation likelihood;
   the gradient of the weighted quantile loss over the observed context (Eqs. 13-14)
   shifts the reverse mean. Trajectory ``s`` uses quantile level ``s / (S + 1)``.

Independent implementation from the paper and the official repository (no license)
at the revision pinned in the model card; no official source was copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.empirical_quantiles import empirical_quantiles
from tsflab.models._inputs import check_window

# Floors used by the method (official constants).
SCALE_FLOOR = 1e-5  # mean-absolute context scale
SCALED_CLIP = 10.0  # scaled values are clipped to [-10, 10]
PRECISION_FLOOR = 1e-6  # Eqs. 8-9
WEIGHT_FLOOR = 1e-8  # Eq. 7


# ---------------------------------------------------------------------------
# Diffusion-step embedding and DDPM schedule
# ---------------------------------------------------------------------------


def step_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer steps ``[B] -> [B, dim]`` (sin half, then cos half)."""
    half = dim // 2
    exponent = math.log(10000.0) / (half - 1)
    frequencies = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -exponent)
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def ddpm_schedule(steps: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    """Linear-beta DDPM coefficients indexed by the 0-based step ``t``."""
    betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])
    table = {
        "betas": betas,
        "sqrt_recip_alphas": alphas.rsqrt(),
        "sqrt_alpha_bar": alpha_bar.sqrt(),
        "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
        # sigma_t^2 of the reverse step (Eq. A.2); zero at t = 0.
        "posterior_variance": betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
    }
    return {name: value.float() for name, value in table.items()}


# ---------------------------------------------------------------------------
# S4 (normal plus low-rank, HiPPO-LegS) layer used by the backbone (App. B.1)
# ---------------------------------------------------------------------------


def hippo_legs_nplr(state_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagonal-plus-low-rank form of HiPPO-LegS, keeping one of each conjugate pair.

    ``A_nk = -sqrt(2n+1) sqrt(2k+1)`` below the diagonal and ``-(n+1)`` on it,
    ``B_n = sqrt(2n+1)``, rank-one ``P_n = sqrt(n + 1/2)``. ``A + P P^T`` is
    ``-1/2 I`` plus a skew-symmetric matrix, so it is unitarily diagonalised as
    ``V diag(w) V^*`` with ``Re(w) = -1/2``. Returns ``(w, P~, B~, V)`` for the
    ``state_size // 2`` eigenvalues with the most negative imaginary parts, with
    ``P~ = V^* P`` and ``B~ = V^* B``.
    """
    n = torch.arange(state_size, dtype=torch.float64)
    root = (2.0 * n + 1.0).sqrt()
    a = -(root[:, None] * root[None, :]) * (n[:, None] > n[None, :])
    a = a - torch.diag(n + 1.0)
    p = (n + 0.5).sqrt()
    normal = a + p[:, None] * p[None, :]
    real = torch.diagonal(normal).mean()
    skew = normal - real * torch.eye(state_size, dtype=torch.float64)
    imaginary, vectors = torch.linalg.eigh(-1j * skew.to(torch.complex128))
    order = torch.argsort(imaginary)[: state_size // 2]
    w = real + 1j * imaginary[order]
    vectors = vectors[:, order]
    inverse = vectors.conj().transpose(-1, -2)
    b_tilde = inverse @ root.to(torch.complex128)
    p_tilde = inverse @ p.to(torch.complex128)
    return (
        w.to(torch.complex64),
        p_tilde.to(torch.complex64),
        b_tilde.to(torch.complex64),
        vectors.to(torch.complex64),
    )


def _with_conjugates(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.conj()], dim=-1)


class NPLRKernel(nn.Module):
    """Convolution kernel of a bilinear-discretised DPLR state space (full S4).

    ``A = diag(w) - P P^*`` with ``w = -exp(log_neg_real) + i w_imag``, per-feature
    step ``dt = exp(log_dt)`` and ``channels`` output maps per feature. The kernel's
    generating function on the roots of unity is evaluated by Cauchy sums over
    conjugate pairs with a Woodbury correction for the rank-one term. ``C`` is
    replaced once by ``C (I - dA^L)`` for the first requested length (and by
    ``C (I + dA^L)`` whenever the length doubles), so the inverse FFT gives the
    truncated kernel ``K_l = C dA^l dB`` for ``l < L``.
    """

    def __init__(
        self, features: int, state_size: int, channels: int, dt_min: float = 1e-3, dt_max: float = 1e-1
    ) -> None:
        super().__init__()
        w, p, b, _ = hippo_legs_nplr(state_size)
        half = state_size // 2
        self.features, self.half = features, half
        log_dt = torch.rand(features) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_neg_real = nn.Parameter(torch.log(-w.real.clamp(max=-1e-3)).repeat(features, 1))
        self.w_imag = nn.Parameter(w.imag.repeat(features, 1))
        self.B = nn.Parameter(torch.view_as_real(b.repeat(1, features, 1)).contiguous())
        self.P = nn.Parameter(torch.view_as_real(p.repeat(1, features, 1)).contiguous())
        self.C = nn.Parameter(torch.view_as_real(torch.randn(channels, features, half, dtype=torch.cfloat)))
        self.register_buffer("kernel_length", torch.tensor(0, dtype=torch.long))

    def diagonal(self) -> torch.Tensor:
        return -torch.exp(self.log_neg_real) + 1j * self.w_imag

    def discrete_transition(self) -> torch.Tensor:
        """Dense bilinear ``dA = (2/dt - A)^-1 (2/dt + A)`` on the full conjugate state ``[H, N, N]``."""
        w = _with_conjugates(self.diagonal())
        p = _with_conjugates(torch.view_as_complex(self.P)[0])
        a = torch.diag_embed(w) - p[:, :, None] * p.conj()[:, None, :]
        dt = torch.exp(self.log_dt).to(a.dtype)[:, None, None]
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        return torch.linalg.solve(2.0 / dt * eye - a, 2.0 / dt * eye + a)

    def discrete_input(self) -> torch.Tensor:
        """Bilinear ``dB = (2/dt - A)^-1 2 B`` on the full conjugate state ``[H, N]``."""
        w = _with_conjugates(self.diagonal())
        p = _with_conjugates(torch.view_as_complex(self.P)[0])
        b = _with_conjugates(torch.view_as_complex(self.B)[0])
        a = torch.diag_embed(w) - p[:, :, None] * p.conj()[:, None, :]
        dt = torch.exp(self.log_dt).to(a.dtype)[:, None, None]
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        return torch.linalg.solve(2.0 / dt * eye - a, 2.0 * b[..., None])[..., 0]

    @torch.no_grad()
    def _prepare_length(self, length: int) -> None:
        while length > int(self.kernel_length):
            current = int(self.kernel_length)
            doubling = current > 0
            power = torch.linalg.matrix_power(self.discrete_transition(), current if doubling else length)
            c = _with_conjugates(torch.view_as_complex(self.C))  # [C, H, N]
            product = torch.einsum("chn,hnm->chm", c, power)
            c = c + product if doubling else c - product
            self.C.copy_(torch.view_as_real(c[..., : self.half].contiguous()))
            self.kernel_length.fill_(2 * current if doubling else length)

    def forward(self, length: int) -> torch.Tensor:
        """Return the real kernel ``[channels, H, length]``."""
        self._prepare_length(length)
        size = int(self.kernel_length)
        dt = torch.exp(self.log_dt)
        w = self.diagonal() * dt[:, None]  # [H, N/2]
        b = torch.view_as_complex(self.B)  # [1, H, N/2]
        p = torch.view_as_complex(self.P)  # [1, H, N/2]
        c = torch.view_as_complex(self.C)  # [C, H, N/2]
        omega = torch.exp(-2j * math.pi * torch.arange(size // 2 + 1, device=w.device) / size)
        z = 2.0 * (1.0 - omega) / (1.0 + omega)  # bilinear map of the roots of unity
        left = torch.cat([b, p], dim=0)  # [2, H, N/2]
        right = torch.cat([c, p.conj()], dim=0)  # [C+1, H, N/2]
        v = left[:, None] * right[None, :]  # [2, C+1, H, N/2]
        # Conjugate-symmetric Cauchy sum: sum_n v/(z - w) + conj(v)/(z - conj(w)).
        numerator = 2.0 * (z * v.real[..., None] - (v * w.conj()).real[..., None])
        denominator = (z - w[..., None]) * (z - w.conj()[..., None])  # [H, N/2, F]
        r = (numerator / denominator).sum(dim=-2) * dt[:, None]  # [2, C+1, H, F]
        k_f = r[0, :-1] - r[0, -1:] * r[1, :-1] / (1.0 + r[1, -1:])  # rank-one Woodbury
        k_f = k_f * 2.0 / (1.0 + omega)
        return torch.fft.irfft(k_f, n=size)[..., :length]


class S4(nn.Module):
    """Bidirectional S4 layer on ``[B, H, L]``: kernel conv + ``D`` skip, GELU, dropout, 1x1 conv."""

    def __init__(self, features: int, state_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.kernel = NPLRKernel(features, state_size, channels=2)
        self.D = nn.Parameter(torch.randn(1, features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output = nn.Conv1d(features, features, kernel_size=1)

    @staticmethod
    def two_sided_kernel(forward_kernel: torch.Tensor, backward_kernel: torch.Tensor) -> torch.Tensor:
        """Causal kernel on ``[0, L)``, flipped anti-causal kernel on ``[L, 2L)`` (circular length ``2L``)."""
        length = forward_kernel.shape[-1]
        return F.pad(forward_kernel, (0, length)) + F.pad(backward_kernel.flip(-1), (length, 0))

    @classmethod
    def convolve(cls, u: torch.Tensor, forward_kernel: torch.Tensor, backward_kernel: torch.Tensor):
        """``y_t = sum_{s<=t} k^f_{t-s} u_s + sum_{s>t} k^b_{s-t-1} u_s`` via a length-``2L`` FFT."""
        length = u.shape[-1]
        kernel = cls.two_sided_kernel(forward_kernel, backward_kernel)
        return torch.fft.irfft(torch.fft.rfft(u, n=2 * length) * torch.fft.rfft(kernel, n=2 * length), n=2 * length)[
            ..., :length
        ]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        forward_kernel, backward_kernel = self.kernel(u.shape[-1])
        y = self.convolve(u, forward_kernel, backward_kernel)
        y = y + u * self.D[0][None, :, None]
        return self.output(self.dropout(F.gelu(y)))


class S4ResidualBlock(nn.Module):
    """Step-conditioned S4 block: pre-LayerNorm S4 residual, gated tanh*sigmoid, two 1x1 heads."""

    def __init__(self, hidden: int, state_size: int, dropout: float) -> None:
        super().__init__()
        self.step_projection = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.s4 = S4(hidden, state_size, dropout)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        self.residual_head = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.skip_head = nn.Conv1d(hidden, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x + self.step_projection(step)[:, :, None]
        h = h + self.dropout(self.s4(self.norm(h.transpose(1, 2)).transpose(1, 2)))
        gated = torch.tanh(h) * torch.sigmoid(h)
        return self.residual_head(gated) + x, self.skip_head(gated)


class S4NoiseBackbone(nn.Module):
    """Unconditional noise predictor ``eps_theta(x_t, t)`` on ``[B, D, L]`` (App. B.1, Fig. B.1)."""

    def __init__(self, channels: int, hidden: int, blocks: int, step_dim: int, state_size: int, dropout: float) -> None:
        super().__init__()
        self.step_dim = step_dim
        self.input_map = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU())
        self.step_map = nn.Sequential(nn.Linear(step_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.blocks = nn.ModuleList(S4ResidualBlock(hidden, state_size, dropout) for _ in range(blocks))
        self.output_map = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, channels))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_map(x_t.transpose(1, 2)).transpose(1, 2)  # [B, hidden, L]
        step = self.step_map(step_embedding(t, self.step_dim))
        skip = torch.zeros_like(h)
        for block in self.blocks:
            h, block_skip = block(h, step)
            skip = skip + block_skip
        # Global skip sum -> output head -> global residual add of the input state.
        return self.output_map(skip.transpose(1, 2)).transpose(1, 2) + x_t


class StateAwarePolicyNetwork(nn.Module):
    """``g_phi([x_t ; x0_hat], t) -> Z_t`` (Sec. 4.1, Fig. 2): relative log-precision ``[B, D, L]``."""

    def __init__(self, channels: int, hidden: int, step_dim: int) -> None:
        super().__init__()
        self.step_dim = step_dim
        self.step_map = nn.Sequential(nn.Linear(step_dim, step_dim), nn.SiLU())
        self.encoder = nn.Sequential(
            nn.Conv1d(2 * channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden + step_dim, hidden, kernel_size=1), nn.SiLU(), nn.Conv1d(hidden, channels, kernel_size=1)
        )

    def forward(self, x_t: torch.Tensor, x0_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(torch.cat([x_t, x0_hat], dim=1))  # S_t = [x_t ; x0_hat], 2D channels
        step = self.step_map(step_embedding(t, self.step_dim))[:, :, None].expand(-1, -1, hidden.shape[-1])
        return self.head(torch.cat([hidden, step], dim=1))  # Eq. 6

    @staticmethod
    def guidance_matrix(scores: torch.Tensor) -> torch.Tensor:
        """Eq. 7: ``A_t = exp(Z_t) / (mean_{i,j} exp(Z_t) + eps)``, global mean about one."""
        weights = scores.exp()
        return weights / (weights.mean(dim=(1, 2), keepdim=True) + WEIGHT_FLOOR)


def proxy_precision_target(x0_hat: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Eqs. 8-9: log-precision z-scored per sample over variables and time, clipped to [-3, 3]."""
    log_precision = -torch.log((x0_hat - x0).square() + PRECISION_FLOOR)
    mean = log_precision.mean(dim=(1, 2), keepdim=True)
    std = log_precision.std(dim=(1, 2), keepdim=True)
    return ((log_precision - mean) / (std + PRECISION_FLOOR)).clamp(-3.0, 3.0)


def pinball(residual: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``rho_kappa(e) = max(kappa e, (kappa - 1) e)`` with ``e = y - x0_hat``."""
    return torch.maximum(kappa * residual, (kappa - 1.0) * residual)


class Model(nn.Module):
    """DynG-Diff with a frozen unconditional S4 backbone and a state-aware policy network.

    ``forward`` returns the empirical quantiles ``[B, pred_len, enc_in, K]`` of
    ``num_samples`` guided reverse-diffusion trajectories.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        quantile_levels: list[float],
        hidden_dim: int = 64,
        num_blocks: int = 3,
        step_embedding_dim: int = 128,
        state_size: int = 128,
        dropout: float = 0.0,
        diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        policy_hidden: int = 64,
        policy_step_dim: int = 32,
        guidance_scale: float = 3.0,
        num_samples: int = 100,
        sample_batch_size: int = 100,
        backbone_epochs: int = 300,
        backbone_batches_per_epoch: int = 100,
        backbone_lr: float = 1e-3,
        backbone_grad_clip: float = 1.0,
        backbone_lr_patience: int = 10,
    ) -> None:
        super().__init__()
        levels = list(quantile_levels)
        if not levels or any(not 0.0 < q < 1.0 for q in levels) or levels != sorted(set(levels)):
            raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
        self.output_type = "quantile"
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.num_steps = diffusion_steps
        self.guidance_scale = guidance_scale
        self.num_samples, self.sample_batch_size = num_samples, sample_batch_size
        self.backbone_epochs = backbone_epochs
        self.backbone_batches_per_epoch = backbone_batches_per_epoch
        self.backbone_lr, self.backbone_grad_clip = backbone_lr, backbone_grad_clip
        self.backbone_lr_patience = backbone_lr_patience
        self.backbone = S4NoiseBackbone(enc_in, hidden_dim, num_blocks, step_embedding_dim, state_size, dropout)
        self.policy = StateAwarePolicyNetwork(enc_in, policy_hidden, policy_step_dim)
        for name, value in ddpm_schedule(diffusion_steps, beta_start, beta_end).items():
            self.register_buffer(name, value, persistent=False)
        self.register_buffer("quantile_levels", torch.tensor(levels), persistent=False)
        self.register_buffer("backbone_frozen", torch.tensor(False))

    # -- shared pieces --------------------------------------------------------
    def _at(self, name: str, t: torch.Tensor) -> torch.Tensor:
        return getattr(self, name)[t].view(-1, 1, 1)

    @staticmethod
    def mean_scale(context: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-variable mean absolute context value ``[B, 1, D]`` (official scaler)."""
        return context.abs().mean(dim=1, keepdim=True) + SCALE_FLOOR

    def scaled_window(self, context: torch.Tensor, future: torch.Tensor | None = None):
        """Clipped scaled window ``[B, D, seq_len + pred_len]`` (zero future if unknown) and its scale."""
        scale = self.mean_scale(context)
        if future is None:
            window = F.pad(context, (0, 0, 0, self.pred_len))
        else:
            window = torch.cat([context, future], dim=1)
        return (window / scale).clamp(-SCALED_CLIP, SCALED_CLIP).transpose(1, 2), scale

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._at("sqrt_alpha_bar", t) * x0 + self._at("sqrt_one_minus_alpha_bar", t) * noise

    def one_step_estimate(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Eq. 3: ``x0_hat = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t)``."""
        return (x_t - self._at("sqrt_one_minus_alpha_bar", t) * noise) / self._at("sqrt_alpha_bar", t)

    def train(self, mode: bool = True):
        super().train(mode)
        if bool(self.backbone_frozen):
            self.backbone.eval()
        return self

    # -- stage 1: unconditional backbone --------------------------------------
    def backbone_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Eq. 2: ``|| eps - eps_theta(x_t, t) ||^2`` with uniform ``t``."""
        t = torch.randint(0, self.num_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(self.backbone(self.q_sample(x0, t, noise), t), noise)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None  # drop stage-1 gradients; the frozen trunk keeps none
        self.backbone_frozen.fill_(True)
        self.backbone.eval()

    def pretrain(self, train_loader, device) -> None:
        """Stage 1 (App. B.1): fit the backbone on training windows, then freeze it.

        ``backbone_epochs`` epochs of ``backbone_batches_per_epoch`` batches (the
        loader is cycled), Adam, gradient-norm clipping and ReduceLROnPlateau on the
        epoch-mean training loss, as in the official backbone script.
        """
        if bool(self.backbone_frozen):
            self.freeze_backbone()
            return
        self.to(device)
        self.backbone.train()
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.backbone_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.backbone_lr_patience
        )
        batches = iter(train_loader)
        for _ in range(self.backbone_epochs):
            total = 0.0
            for _ in range(self.backbone_batches_per_epoch):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(train_loader)
                    batch = next(batches)
                x, y = batch[0].float().to(device), batch[1].float().to(device)
                x0, _ = self.scaled_window(x, y[:, -self.pred_len :])
                optimizer.zero_grad()
                loss = self.backbone_loss(x0)
                loss.backward()
                if self.backbone_grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.backbone.parameters(), self.backbone_grad_clip)
                optimizer.step()
                total += float(loss.detach())
            scheduler.step(total / self.backbone_batches_per_epoch)
        self.freeze_backbone()

    # -- stage 2: state-aware policy network ----------------------------------
    def policy_loss(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Eq. 10: MSE between the policy score ``Z_t`` and the proxy target of Eq. 9."""
        x0, _ = self.scaled_window(context, future)
        t = torch.randint(0, self.num_steps, (x0.shape[0],), device=x0.device)
        with torch.no_grad():
            x_t = self.q_sample(x0, t, torch.randn_like(x0))
            x0_hat = self.one_step_estimate(x_t, t, self.backbone(x_t, t))
            target = proxy_precision_target(x0_hat, x0)
        return F.mse_loss(self.policy(x_t, x0_hat, t), target)

    # -- stage 3: guided reverse diffusion ------------------------------------
    def guidance_score(self, x_t, t, observation, mask, kappa):
        """Return ``(eps_theta(x_t, t), -grad_x L_guide)`` with ``A_t`` detached (Eqs. 7, 13-14)."""
        with torch.enable_grad():
            probe = x_t.detach().requires_grad_(True)
            noise = self.backbone(probe, t)
            x0_hat = self.one_step_estimate(probe, t, noise)
            with torch.no_grad():
                weights = self.policy.guidance_matrix(self.policy(probe.detach(), x0_hat.detach(), t))
            energy = (mask * weights * pinball(observation - x0_hat, kappa)).sum()
            (gradient,) = torch.autograd.grad(energy, probe)
        return noise.detach(), -gradient

    def reverse_step(self, x_t, step: int, observation, mask, kappa):
        """DDPM step (Eq. A.2) whose mean is shifted by ``s sigma_t^2 grad log p(y_obs | x_t)``."""
        t = torch.full((x_t.shape[0],), step, device=x_t.device, dtype=torch.long)
        noise, score = self.guidance_score(x_t, t, observation, mask, kappa)
        mean = self._at("sqrt_recip_alphas", t) * (
            x_t - self._at("betas", t) * noise / self._at("sqrt_one_minus_alpha_bar", t)
        )
        variance = self._at("posterior_variance", t)
        mean = mean + self.guidance_scale * variance * score
        if step == 0:
            return mean
        return mean + variance.sqrt() * torch.randn_like(x_t)

    @staticmethod
    def trajectory_levels(count: int, offset: int, total: int, device=None) -> torch.Tensor:
        """``kappa_s = s / (S + 1)`` for trajectories ``s = offset + 1 .. offset + count``."""
        index = torch.arange(offset + 1, offset + count + 1, device=device, dtype=torch.float32)
        return index / (total + 1)

    def sample(self, x_enc: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """Guided trajectories mapped back to the input scale: ``[S, B, pred_len, D]``."""
        total = num_samples or self.num_samples
        observation, scale = self.scaled_window(x_enc)
        mask = torch.zeros_like(observation)
        mask[..., : self.seq_len] = 1.0
        batch = x_enc.shape[0]
        chunks, offset = [], 0
        while offset < total:
            count = min(self.sample_batch_size, total - offset)
            kappa = self.trajectory_levels(count, offset, total, x_enc.device)
            kappa = kappa.repeat_interleave(batch).view(-1, 1, 1)
            obs = observation.repeat(count, 1, 1)
            obs_mask = mask.repeat(count, 1, 1)
            x = torch.randn_like(obs)
            for step in reversed(range(self.num_steps)):
                x = self.reverse_step(x, step, obs, obs_mask, kappa)
            future = x.detach()[..., self.seq_len :].transpose(1, 2).reshape(count, batch, self.pred_len, -1)
            chunks.append(future * scale[None])
            offset += count
        return torch.cat(chunks, dim=0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        samples = self.sample(x_enc)
        return empirical_quantiles(samples, self.quantile_levels.to(samples.dtype))

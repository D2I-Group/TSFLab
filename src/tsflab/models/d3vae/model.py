"""Independent PyTorch implementation of D3VAE (Li et al., NeurIPS 2022).

Written from the paper (arXiv 2301.03028, Sec. 2 and Appendix C) and checked
against the pinned official PaddlePaddle code; no official source is copied.
The forecaster embeds the history, encodes it with a two-layer GRU, feeds the
concatenated representation to a bidirectional hierarchical VAE (BVAE, an
NVAE-style encoder/decoder over a ``[time, feature]`` grid), and cleans the
generated mean with one gradient step of a learned energy (Eq. 11).  Training
uses the coupled diffusion of input and target (Eq. 3-6) and the loss of Eq. 14.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import PositionalEmbedding, TokenEmbedding
from tsflab.models._components.marks import adapt_tslib_marks
from tsflab.models._inputs import check_window


def linear_beta_schedule(beta_start: float, beta_end: float, steps: int) -> torch.Tensor:
    """Uniformly increasing variance schedule beta_1..beta_T (Sec. 2.2.1)."""
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)


def soft_clamp(x: torch.Tensor, bound: float) -> torch.Tensor:
    """Smooth clamp ``bound * tanh(x / bound)`` applied to Gaussian parameters."""
    return bound * torch.tanh(x / bound)


def total_correlation(
    z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Batch-normalized total-correlation score of one latent group (Eq. 12).

    ``z``, ``mu`` and ``log_var`` are ``[B, m, H, W]`` with the ``m`` latent
    channels as the factors.  Following the official estimator, the joint
    log-density aggregates every grid cell of a sample before a log-sum-exp over
    factors, the factorized density takes the log-sum-exp over factors per cell,
    and their difference is min-max normalized over the batch before averaging.
    """
    log_density = -0.5 * (math.log(2 * math.pi) + log_var) - 0.5 * (z - mu) ** 2 * torch.exp(
        -log_var
    )
    log_joint = torch.logsumexp(log_density.sum(dim=(2, 3)), dim=1)
    log_factorized = torch.logsumexp(log_density, dim=1).sum(dim=(1, 2))
    score = log_joint - log_factorized
    spread = (score.max() - score.min()).clamp_min(eps)
    return ((score - score.min()) / spread).mean()


class CalendarEmbedding(nn.Module):
    """Learned month/day/weekday/hour(/quarter-hour) tables, concatenated then projected."""

    def __init__(self, d_model: int, freq: str) -> None:
        super().__init__()
        self.freq = freq
        self.month = nn.Embedding(13, d_model)
        self.day = nn.Embedding(32, d_model)
        self.weekday = nn.Embedding(7, d_model)
        self.hour = nn.Embedding(24, d_model)
        self.minute = nn.Embedding(4, d_model) if freq == "t" else None
        tables = 5 if freq == "t" else 4
        self.projection = nn.Linear(tables * d_model, d_model)

    def forward(self, marks: torch.Tensor) -> torch.Tensor:
        raw = marks.shape[-1] == 6
        calendar = adapt_tslib_marks(marks, embed_type="learned", freq=self.freq).long()
        tables = (self.hour, self.weekday, self.day, self.month)
        parts = [table(calendar[..., column]) for table, column in zip(tables, (3, 2, 1, 0))]
        if self.minute is not None:
            minute = calendar[..., 4] // 15 if raw else calendar[..., 4]
            parts.insert(0, self.minute(minute))
        return self.projection(torch.cat(parts, dim=-1))


class WeightNormConv2d(nn.Conv2d):
    """Conv2d with NVAE weight normalization ``exp(log_gain) * w / ||w||``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        norm = self.weight.detach().flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        self.log_gain = nn.Parameter(torch.log(norm + 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.weight.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        weight = torch.exp(self.log_gain) * self.weight / (norm + 1e-5)
        return self._conv_forward(x, weight, self.bias)


def _batch_norm(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.05)


class BNSwishConv(nn.Module):
    """BatchNorm, Swish, then a weight-normalized 3x3 convolution (stride 1 or 2)."""

    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.norm = _batch_norm(c_in)
        self.conv = WeightNormConv2d(c_in, c_out, 3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.silu(self.norm(x)))


class InvertedResidual(nn.Module):
    """MobileNet-style op: [upsample], BN, 1x1 expand, depthwise kxk, 1x1 project, BN."""

    def __init__(self, c_in: int, c_out: int, expansion: int, kernel: int, upsample: bool) -> None:
        super().__init__()
        hidden = c_in * expansion
        self.upsample = upsample
        self.layers = nn.Sequential(
            _batch_norm(c_in),
            nn.Conv2d(c_in, hidden, 1), _batch_norm(hidden), nn.SiLU(),  # expand
            nn.Conv2d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden),  # depthwise
            _batch_norm(hidden), nn.SiLU(),
            nn.Conv2d(hidden, c_out, 1), _batch_norm(c_out),  # project
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.layers(x)


class FactorizedReduce(nn.Module):
    """Stride-2 skip path: Swish, four strided 1x1 convs at row offsets 0/1, concatenated."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        quarter = c_out // 4
        widths = (quarter, quarter, quarter, c_out - 3 * quarter)
        self.convs = nn.ModuleList(WeightNormConv2d(c_in, w, 1, stride=2) for w in widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(x)
        shifted = x[:, :, 1:]
        return torch.cat(
            [conv(x if index % 2 == 0 else shifted) for index, conv in enumerate(self.convs)],
            dim=1,
        )


class UpsampleSkip(nn.Module):
    """Upsampling skip path: bilinear x2 then a 1x1 convolution halving channels."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = WeightNormConv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


# Cell kind -> (skip path or None for identity, operations from (c_in, c_out)).
_CELL_KINDS = {
    "normal_pre": (None, lambda a, b: [BNSwishConv(a, b, 1), BNSwishConv(b, b, 1)]),
    "normal_enc": (None, lambda a, b: [BNSwishConv(a, b, 1), BNSwishConv(b, b, 1)]),
    "down_pre": (FactorizedReduce, lambda a, b: [BNSwishConv(a, b, 2), BNSwishConv(b, b, 1)]),
    "normal_dec": (None, lambda a, b: [InvertedResidual(a, b, 6, 5, upsample=False)]),
    "up_post": (UpsampleSkip, lambda a, b: [InvertedResidual(a, b, 3, 5, upsample=True)]),
    "normal_post": (None, lambda a, b: [InvertedResidual(a, b, 3, 5, upsample=False)]),
}


class Cell(nn.Module):
    """Residual cell ``skip(s) + 0.1 * ops(s)`` of the BVAE towers."""

    def __init__(self, c_in: int, c_out: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        skip, ops = _CELL_KINDS[kind]
        self.skip: nn.Module = nn.Identity() if skip is None else skip(c_in, c_out)
        self.ops = nn.Sequential(*ops(c_in, c_out))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.skip(s) + 0.1 * self.ops(s)


def _pre_process_cells(channels: int, num_blocks: int, cells: int) -> nn.Sequential:
    """Encoder preprocessing; each block ends with a stride-2 cell doubling the channels."""
    pre = []
    for _ in range(num_blocks):
        pre += [Cell(channels, channels, "normal_pre") for _ in range(cells - 1)]
        pre.append(Cell(channels, 2 * channels, "down_pre"))
        channels *= 2
    return nn.Sequential(*pre)


def _post_process_cells(channels: int, num_blocks: int, cells: int) -> nn.Sequential:
    """Decoder postprocessing; each block starts with an upsampling cell halving the channels."""
    post = []
    for _ in range(num_blocks):
        post.append(Cell(channels, channels // 2, "up_post"))
        channels //= 2
        post += [Cell(channels, channels, "normal_post") for _ in range(cells - 1)]
    return nn.Sequential(*post)


class BidirectionalVAE(nn.Module):
    """BVAE over a ``[B, 1, time, width]`` grid with ``groups`` latent variables.

    The bottom-up encoder (stem, preprocessing cells, encoder tower) feeds the
    top-down decoder through encoder combiners; each latent group ``z_i`` is
    drawn from ``p_phi(z_i | z_<i, X)`` and merged by a decoder combiner, then
    postprocessing cells and a projection over the width axis emit the Gaussian
    parameters of the target ``[B, 1, time, target_dim]``.
    """

    def __init__(
        self,
        time_len: int, width: int, target_dim: int,
        num_channels_enc: int, num_channels_dec: int, num_blocks: int,
        num_preprocess_cells: int, num_postprocess_cells: int,
        groups: int, latent_per_group: int, clamp: float,
    ) -> None:
        super().__init__()
        self.width, self.clamp = width, clamp
        self.stem = WeightNormConv2d(1, num_channels_enc, 3, padding=1)
        self.pre_process = _pre_process_cells(num_channels_enc, num_blocks, num_preprocess_cells)
        enc_c, dec_c = num_channels_enc * 2**num_blocks, num_channels_dec * 2**num_blocks
        self.enc_cells = nn.ModuleList(Cell(enc_c, enc_c, "normal_enc") for _ in range(groups))
        # Encoder combiner: x_enc + conv1x1(decoder state).
        self.enc_combiners = nn.ModuleList(
            WeightNormConv2d(dec_c, enc_c, 1) for _ in range(groups - 1)
        )
        self.enc0 = nn.Sequential(nn.ELU(), WeightNormConv2d(enc_c, enc_c, 1), nn.ELU())
        self.enc_samplers = nn.ModuleList(
            WeightNormConv2d(enc_c, 2 * latent_per_group, 3, padding=1) for _ in range(groups)
        )
        self.dec_cells = nn.ModuleList(Cell(dec_c, dec_c, "normal_dec") for _ in range(groups - 1))
        # Decoder combiner: conv1x1(concat(decoder state, z)).
        self.dec_combiners = nn.ModuleList(
            WeightNormConv2d(dec_c + latent_per_group, dec_c, 1) for _ in range(groups)
        )
        latent_width = -(-width // 2**num_blocks)  # ceil-halved once per block
        self.prior_ftr0 = nn.Parameter(torch.rand(dec_c, time_len // 2**num_blocks, latent_width))
        self.post_process = _post_process_cells(dec_c, num_blocks, num_postprocess_cells)
        self.image_conditional = nn.Sequential(
            nn.ELU(), WeightNormConv2d(num_channels_dec, 2, 3, padding=1)
        )
        self.projection = nn.Linear(width, target_dim)

    def _posterior(self, features: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.enc_samplers[index](features).chunk(2, dim=1)
        mu, log_sigma = (soft_clamp(part, self.clamp) for part in params)
        z = mu + torch.exp(log_sigma) * torch.randn_like(mu)
        # The official TC estimator reads the second chunk as a log-variance.
        return z, total_correlation(z, mu, log_sigma)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return target mean, target log-scale ``[B, 1, T, D]`` and the mean TC (Eq. 13)."""
        s = self.pre_process(self.stem(2 * x - 1.0))
        skips = []
        for cell in self.enc_cells:
            s = cell(s)
            skips.append(s)
        skips = skips[-2::-1]  # all but the last encoder state, top-down
        combiners = list(self.enc_combiners)[::-1]

        z, tc = self._posterior(self.enc0(s), 0)
        tc_terms = [tc]
        state = self.prior_ftr0.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        state = self.dec_combiners[0](torch.cat([state, z], dim=1))
        for index in range(1, len(self.dec_combiners)):
            state = self.dec_cells[index - 1](state)
            features = skips[index - 1] + combiners[index - 1](state)
            z, tc = self._posterior(features, index)
            tc_terms.append(tc)
            state = self.dec_combiners[index](torch.cat([state, z], dim=1))
        params = self.image_conditional(self.post_process(state))
        params = self.projection(params[..., -self.width :])
        mu, log_sigma = (soft_clamp(part, self.clamp) for part in params.chunk(2, dim=1))
        return mu, log_sigma, torch.stack(tc_terms).mean()


class EnergyResidualBlock(nn.Module):
    """``conv1x1(x) + conv3x3(ELU(conv3x3(ELU(x))))`` at constant resolution."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.shortcut = nn.Conv2d(c_in, c_out, 1)
        self.conv1 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.conv2 = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.conv2(F.elu(self.conv1(F.elu(x))))


class EnergyNetwork(nn.Module):
    """Energy ``E(Y; zeta)`` with a quadratic read-out ``l1(h) l2(h) + lq(h^2)``."""

    def __init__(self, channels: int, time_len: int, target_dim: int) -> None:
        super().__init__()
        widths = [factor * channels for factor in (1, 2, 2, 4, 4, 8, 8)]
        self.conv_in = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *(EnergyResidualBlock(a, b) for a, b in zip(widths[:-1], widths[1:]))
        )
        features = 8 * channels * time_len * target_dim
        self.linear1 = nn.Linear(features, 1)
        self.linear2 = nn.Linear(features, 1)
        self.quadratic = nn.Linear(features, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.conv_in(y)).flatten(1)
        energy = self.linear1(h) * self.linear2(h) + self.quadratic(h * h)
        return energy.squeeze(-1)


class Model(nn.Module):
    """D3VAE: coupled diffusion, BVAE generation, energy denoising, disentanglement."""

    output_type = "point"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        freq: str = "h",
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        diff_steps: int = 100,
        beta_start: float = 0.0,
        beta_end: float = 0.01,
        target_scale: float = 0.1,
        num_channels_enc: int = 32,
        num_channels_dec: int = 32,
        num_blocks: int = 1,
        num_preprocess_cells: int = 3,
        num_postprocess_cells: int = 2,
        groups_per_scale: int = 2,
        latent_per_group: int = 8,
        score_channels: int = 64,
        denoise_step: float = 0.001,
        clamp: float = 1.0,
        psi: float = 0.5,
        lambda_dsm: float = 1.0,
        gamma: float = 0.01,
    ) -> None:
        super().__init__()
        checks = (
            (features not in {"M", "S", "MS"}, "features must be M, S, or MS"),
            (pred_len % 2**num_blocks, "pred_len must be divisible by 2 ** num_blocks"),
        )
        for failed, message in checks:
            if failed:
                raise ValueError(message)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.features = features
        self.target_dim = 1 if features == "MS" else enc_in
        self.diff_steps, self.denoise_step = diff_steps, denoise_step
        self.psi, self.lambda_dsm, self.gamma = psi, lambda_dsm, gamma

        self.value_embedding = TokenEmbedding(enc_in, embedding_dim)
        self.position_embedding = PositionalEmbedding(embedding_dim)
        self.calendar_embedding = CalendarEmbedding(embedding_dim, freq)
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            embedding_dim, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        # Local adapter: the BVAE grid is horizon-long; identity when seq_len == pred_len.
        self.time_map = nn.Linear(seq_len, pred_len) if seq_len != pred_len else None
        width = embedding_dim + hidden_size
        self.bvae = BidirectionalVAE(
            pred_len, width, self.target_dim, num_channels_enc, num_channels_dec, num_blocks,
            num_preprocess_cells, num_postprocess_cells, groups_per_scale, latent_per_group, clamp,
        )
        self.energy = EnergyNetwork(score_channels, pred_len, self.target_dim)

        betas = linear_beta_schedule(beta_start, beta_end, diff_steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        target_cumprod = torch.cumprod(1.0 - target_scale * betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)
        self.register_buffer("target_alphas_cumprod", target_cumprod.float(), persistent=False)
        # Multi-scale DSM noise levels sigma_t = 1 - abar_t, weight l(sigma_t) = sigma_t (Eq. 10).
        self.register_buffer("dsm_sigmas", (1.0 - alphas_cumprod).float(), persistent=False)

    # ------------------------------------------------------------------ inputs
    def representation(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """``X_input = concat(GRU(E(X)), E(X))`` mapped to the horizon: ``[B, pred_len, E+H]``."""
        check_window(x_enc, self.seq_len, self.enc_in)
        embedded = self.value_embedding(x_enc) + self.position_embedding(x_enc)
        if x_mark_enc is not None:
            if x_mark_enc.ndim != 3 or x_mark_enc.shape[:2] != x_enc.shape[:2]:
                raise ValueError("x_mark_enc must have shape [batch, seq_len, marks]")
            embedded = embedded + self.calendar_embedding(x_mark_enc)
        embedded = self.embedding_dropout(embedded)
        recurrent, _ = self.rnn(embedded)
        features = torch.cat([recurrent, embedded], dim=-1)
        if self.time_map is not None:
            features = self.time_map(features.transpose(1, 2)).transpose(1, 2)
        return features

    # --------------------------------------------------------------- diffusion
    @staticmethod
    def _at(schedule: torch.Tensor, step: torch.Tensor, ndim: int) -> torch.Tensor:
        return schedule[step].view(-1, *([1] * (ndim - 1)))

    def diffuse(
        self, schedule: torch.Tensor, x: torch.Tensor, step: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """``X(t) = sqrt(abar_t) X + sqrt(1 - abar_t) dX``: the input (Eq. 3-4) with
        ``alphas_cumprod`` and the target (Eq. 6, ``beta'_t = omega beta_t``) with
        ``target_alphas_cumprod``."""
        abar = self._at(schedule, step, x.ndim)
        return abar.sqrt() * x + (1.0 - abar).sqrt() * noise

    # ----------------------------------------------------------------- denoise
    def energy_gradient(self, y: torch.Tensor, create_graph: bool) -> torch.Tensor:
        """``grad_Y E(Y; zeta)`` per sample, also under an outer ``no_grad``."""
        with torch.enable_grad():
            probe = y if y.requires_grad else y.detach().requires_grad_(True)
            energy = self.energy(probe).sum()
            (gradient,) = torch.autograd.grad(energy, probe, create_graph=create_graph)
        return gradient

    def forecast_with_uncertainty(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Algorithm 2 / Eq. 11: ``(Y_clean, Y_generated, sigma0^2 grad E)`` as ``[B, H, D]``."""
        features = self.representation(x_enc, x_mark_enc)
        mu, _, _ = self.bvae(features.unsqueeze(1))
        gradient = self.energy_gradient(mu, create_graph=torch.is_grad_enabled())
        uncertainty = self.denoise_step * gradient
        clean = mu - uncertainty
        return clean.squeeze(1), mu.squeeze(1), uncertainty.squeeze(1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        clean, _, _ = self.forecast_with_uncertainty(x_enc, x_mark_enc)
        return clean

    # ---------------------------------------------------------------- training
    def loss_terms(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        y: torch.Tensor,
        step: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Algorithm 1 terms for target ``y`` ``[B, pred_len, target_dim]``."""
        if y.shape != (x_enc.shape[0], self.pred_len, self.target_dim):
            raise ValueError(f"target must have shape [batch, {self.pred_len}, {self.target_dim}]")
        if step is None:
            step = torch.randint(0, self.diff_steps, (x_enc.shape[0],), device=x_enc.device)
        features = self.representation(x_enc, x_mark_enc)
        noisy_input = self.diffuse(self.alphas_cumprod, features, step, torch.randn_like(features))
        target = y.unsqueeze(1)
        noisy_target = self.diffuse(
            self.target_alphas_cumprod, target, step, torch.randn_like(target)
        )
        mu, log_sigma, tc = self.bvae(noisy_input.unsqueeze(1))
        sigma = torch.exp(log_sigma)

        # psi-term: Gaussian negative log-likelihood of the diffused target.
        log_prob = (
            -0.5 * ((noisy_target - mu) / sigma) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        )
        nll = -log_prob.sum(dim=(1, 2, 3)).mean()
        generated = mu + sigma * torch.randn_like(mu)
        mse = F.mse_loss(generated, noisy_target)

        # Eq. 10: multi-scale DSM on a detached generated sample.
        sample = (mu + sigma * torch.randn_like(mu)).detach().requires_grad_(True)
        gradient = self.energy_gradient(sample, create_graph=True)
        residual = target - sample + self.denoise_step * gradient
        weight = self._at(self.dsm_sigmas, step, residual.ndim)
        dsm = (residual.pow(2) * weight).sum(dim=(1, 2, 3)).mean()
        return {"nll": nll, "dsm": dsm, "tc": tc, "mse": mse}

    def training_loss(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None, y: torch.Tensor
    ) -> torch.Tensor:
        """Eq. 14: ``psi * NLL + lambda * DSM + gamma * TC + MSE``."""
        terms = self.loss_terms(x_enc, x_mark_enc, y)
        weighted = self.psi * terms["nll"] + self.lambda_dsm * terms["dsm"]
        return weighted + self.gamma * terms["tc"] + terms["mse"]

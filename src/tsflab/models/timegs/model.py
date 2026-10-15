"""TimeGS: forecasting as 2D Gaussian splatting (Wang et al., KDD 2026).

Independent implementation from the paper (Sections 4.1-4.5, Eqs. 4-17,
Appendix B Eq. 18) with omissions resolved against yixinwang1/TimeGS@b794d5bd
(``models/TimeGS.py``, ``layers/TimeGS_Enc.py``); no upstream code is copied.

Pipeline per channel (channel-independent; only the aggregation weights are
channel-specific):

1. 2D-VFE (Eqs. 4-6): RevIN, left zero-pad, fold into rows of ``fold_width``,
   bilinear upsample to ``(image_height, image_width)``; each of the ``K``
   branches runs its own UNet encoder and a linear map to ``G`` anchor
   features ``Z_k in R^{G x D_h}``.
2. MB-GKG (Eqs. 7-11): a fixed, normalized dictionary of ``M`` elliptically
   truncated Gaussian profiles on a ``rows x draft_len`` grid; softmax basis
   mixing weights ``W_k`` and MLP intensities ``V_k`` for ``P`` components.
3. MP-CCR (Eqs. 12-14): each kernel is cropped or zero-padded to the branch
   period ``psi_k``, flattened row-major, and splatted onto the 1D horizon so
   its centre lands on the anchor position, which wraps the 2D support across
   period boundaries in chronological order.
4. CAA (Eqs. 15-17): channel-wise softmax branch weights ``Gamma [C, K]`` and
   component weights ``Omega [C, P]`` fuse the ``K x P`` renders.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.revin import RevIN


def gaussian_basis_bank(
    l11: Sequence[float],
    l21: Sequence[float],
    l22: Sequence[float],
    coefficient: Sequence[float],
    rows: int,
    width: int,
) -> torch.Tensor:
    """Fixed Gaussian basis bank ``D`` (Eqs. 7-8) as ``[M, rows * width]``.

    One basis per element of ``l11 x l21 x l22 x coefficient`` (in that order).
    Following the official construction, on row offset ``j`` (centred,
    ``j = row - rows // 2``) the profile is centred at column
    ``width // 2 - l21 j / l11``, evaluated as
    ``exp(-coefficient / 2 * [(dx / l11)^2 + ((j - l21 dx / l11) / l22)^2])``,
    truncated to ``|dx| <= sqrt(1 - (l22 j)^2) / l11``, and normalized to unit
    sum over its discrete support (Eq. 8).
    """
    grid = torch.tensor(list(itertools.product(l11, l21, l22, coefficient)), dtype=torch.float32)
    a = grid[:, 0].clamp(min=1e-4).view(-1, 1, 1)
    b = grid[:, 1].view(-1, 1, 1)
    c = grid[:, 2].clamp(min=1e-4).view(-1, 1, 1)
    scale = grid[:, 3].view(-1, 1, 1)
    j = (torch.arange(rows) - rows // 2).view(1, -1, 1).float()
    col = torch.arange(width).view(1, 1, -1).float()
    centre = width // 2 - b * j / a
    half = torch.sqrt(1 - (c * j).square().clamp(max=1)) / a
    support = (half > 0) & (col >= centre - half) & (col <= centre + half)
    dx = col - centre
    quadratic = (dx / a).square() + ((j - b / a * dx) / c).square()
    profile = torch.exp(-0.5 * quadratic * scale) * support
    profile = profile.reshape(grid.shape[0], rows * width)
    return profile / profile.sum(dim=-1, keepdim=True).clamp(min=1e-4)


def render_kernels(kernels: torch.Tensor, period: int, positions: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Chronologically continuous rasterization (Eqs. 12-14).

    ``kernels`` is ``[..., G, P, rows, width]`` and ``positions`` the ``[G]``
    anchor coordinates on the horizon. Rows are centrally cropped to at most
    ``2 ceil(pred_len / period) - 1``; columns are centrally cropped or
    symmetrically zero-padded to ``period``; the kernel is flattened row-major
    and placed so its centre ``(rows // 2, period // 2)`` lands on the anchor.
    Returns ``[..., P, pred_len]`` (the sum over anchors).
    """
    rows, width = kernels.shape[-2:]
    keep = min(rows, math.ceil(pred_len / period) * 2 - 1)
    top = (rows - keep) // 2
    kernels = kernels[..., top : top + keep, :]
    if width >= period:
        left = width // 2 - period // 2
        kernels = kernels[..., left : left + period]
    else:
        left = period // 2 - width // 2
        kernels = F.pad(kernels, (left, period - width - left))
    flat = kernels.flatten(-2)
    length = flat.shape[-1]
    centre = (keep // 2) * period + period // 2
    t = torch.arange(pred_len, device=flat.device)
    index = t.view(1, -1) - positions.view(-1, 1).to(flat.device) + centre  # [G, O]
    inside = ((index >= 0) & (index < length)).to(flat.dtype)
    index = index.clamp(0, length - 1)
    lead = flat.shape[:-1]  # [..., G, P]
    gathered = torch.gather(flat, -1, index.view(-1, 1, pred_len).expand(*lead, pred_len))
    gathered = gathered * inside.view(-1, 1, pred_len)
    return gathered.sum(dim=-3)


class ResidualBlock(nn.Module):
    """Bottleneck block ``x + BN(Conv(Dropout(ReLU(BN(Conv(x))))))`` with 3x3 convs."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class UNetEncoder(nn.Module):
    """UNet-based 2D variation encoder (Eq. 5).

    Stem conv, ``depth`` stride-2 downsampling stages, ``n_blocks`` residual
    blocks, ``depth`` transposed-conv upsampling stages that concatenate the
    matching downsampling output (all but the deepest), and a tanh output conv
    with ``out_channels`` maps at the input resolution.
    """

    def __init__(
        self, out_channels: int, base: int, depth: int, n_blocks: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.stem = nn.Sequential(nn.Conv2d(1, base, kernel_size, padding=pad), nn.BatchNorm2d(base), nn.ReLU())
        self.down = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(base * 2**i, base * 2 ** (i + 1), 3, stride=2, padding=1),
                nn.BatchNorm2d(base * 2 ** (i + 1)),
                nn.ReLU(),
            )
            for i in range(depth)
        )
        bottleneck = base * 2**depth
        self.blocks = nn.Sequential(*(ResidualBlock(bottleneck, dropout) for _ in range(n_blocks)))
        self.up = nn.ModuleList()
        for i in range(depth):
            width = base * 2 ** (depth - i)
            self.up.append(
                nn.Sequential(
                    nn.ConvTranspose2d(width * 2 if i else width, width // 2, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(width // 2),
                    nn.ReLU(),
                )
            )
        self.head = nn.Sequential(nn.Conv2d(base, out_channels, kernel_size, padding=pad), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.down):
            x = stage(x)
            if i < len(self.down) - 1:
                skips.append(x)
        x = self.blocks(x)
        for i, stage in enumerate(self.up):
            if i:
                x = torch.cat([x, skips.pop()], dim=1)
            x = stage(x)
        return self.head(x)


class IntensityMLP(nn.Module):
    """Kernel intensity ``V = MLP(Z)`` (Eq. 10): Linear-GELU-Linear, Xavier weights, zero biases."""

    def __init__(self, hidden: int, components: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, components))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        periods: Sequence[int] = (24, 24, 24),
        fold_width: int = 24,
        image_height: int = 24,
        image_width: int = 24,
        hidden_dim: int = 16,
        conv_dim: int = 8,
        components: int = 2,
        ngf: int = 4,
        n_downsampling: int = 3,
        n_blocks: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.5,
        rows: int = 7,
        draft_len: int = 11,
        ratio: int = 1,
        extend_len: int = 0,
        cholesky1: Sequence[float] = (0.4, 0.8, 1.2),
        cholesky2: Sequence[float] = (0.0, -0.2, 0.2),
        cholesky3: Sequence[float] = (0.4, 0.8, 1.2),
        coefficient: Sequence[float] = (0.5, 1.0),
        temperature: float = 0.1,
        weight_norm: str = "official",
        mse_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.fold_width = fold_width
        self.image_size = (image_height, image_width)
        self.hidden_dim, self.components = hidden_dim, components
        self.temperature, self.weight_norm, self.mse_weight = temperature, weight_norm, mse_weight
        # G anchors cover the horizon at stride `ratio`, extended by `extend_len` on both sides.
        self.anchors = pred_len // ratio + 2 * extend_len
        span = pred_len + 2 * extend_len * ratio
        self.periods = [span if p == 0 or p > span else int(p) for p in periods]
        self.branches = len(self.periods)
        self.rows = min(rows, math.ceil(pred_len / min(self.periods)) * 2 - 1)
        self.draft_len = draft_len
        bank = gaussian_basis_bank(cholesky1, cholesky2, cholesky3, coefficient, self.rows, draft_len)
        self.register_buffer("basis_bank", bank, persistent=False)
        self.num_bases = bank.shape[0]
        # Anchor g of branch k sits at (g - extend_len) * ratio + floor(k * ratio / K).
        offsets = torch.arange(self.branches) * ratio // self.branches
        positions = (torch.arange(self.anchors) - extend_len).view(1, -1) * ratio + offsets.view(-1, 1)
        self.register_buffer("anchor_positions", positions, persistent=False)

        self.revin = RevIN(enc_in, affine=False)
        pixels = image_height * image_width
        self.encoders = nn.ModuleList(
            UNetEncoder(conv_dim, ngf, n_downsampling, n_blocks, kernel_size, dropout) for _ in range(self.branches)
        )
        self.projections = nn.ModuleList(
            nn.Linear(conv_dim * pixels, hidden_dim * self.anchors) for _ in range(self.branches)
        )
        self.intensity = nn.ModuleList(IntensityMLP(hidden_dim, components) for _ in range(self.branches))
        self.mixing = nn.ModuleList(nn.Linear(hidden_dim, components * self.num_bases) for _ in range(self.branches))
        self.branch_weight = nn.Parameter(torch.full((enc_in, self.branches), 1.0 / self.branches))
        self.component_weight = nn.Parameter(torch.full((enc_in, components), 1.0 / components))

    def fold(self, series: torch.Tensor) -> torch.Tensor:
        """Eq. (4): ``[B, C, L]`` -> left zero-pad, fold, bilinear upsample -> ``[B*C, 1, H', W']``."""
        batch, channels, length = series.shape
        pad = -length % self.fold_width
        if pad:
            series = F.pad(series, (pad, 0))
        image = series.reshape(batch * channels, 1, -1, self.fold_width)
        return F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)

    def branch_kernels(self, image: torch.Tensor, batch: int, branch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (5)-(11) for one branch: intensities ``[B, C, G, P]`` and normalized
        composite kernel shapes ``[B, C, G, P, rows * draft_len]``."""
        feats = self.encoders[branch](image).flatten(1)
        z = self.projections[branch](feats).view(batch, -1, self.anchors, self.hidden_dim)
        intensity = self.intensity[branch](z)
        logits = self.mixing[branch](z).view(*z.shape[:3], self.components, self.num_bases)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return intensity, weights @ self.basis_bank

    def aggregation_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (15)-(16): channel-wise softmax over branches and components.

        ``official`` divides each row by its sum before the softmax, as the
        pinned code does; ``paper`` is the plain softmax.
        """
        gamma, omega = self.branch_weight, self.component_weight
        if self.weight_norm == "official":
            gamma = gamma / gamma.sum(dim=-1, keepdim=True)
            omega = omega / omega.sum(dim=-1, keepdim=True)
        return torch.softmax(gamma, dim=-1), torch.softmax(omega, dim=-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.revin(x_enc, "norm")
        series = x.transpose(1, 2)
        batch, channels = series.shape[:2]
        image = self.fold(series)
        renders = []
        for k, period in enumerate(self.periods):
            intensity, shapes = self.branch_kernels(image, batch, k)
            kernels = (shapes * intensity.unsqueeze(-1)).view(
                batch, channels, self.anchors, self.components, self.rows, self.draft_len
            )
            renders.append(render_kernels(kernels, period, self.anchor_positions[k], self.pred_len))
        stacked = torch.stack(renders, dim=2)  # [B, C, K, P, O]
        gamma, omega = self.aggregation_weights()
        weights = gamma.view(1, channels, -1, 1, 1) * omega.view(1, channels, 1, -1, 1)
        forecast = (stacked * weights).sum(dim=(2, 3)).transpose(1, 2)
        return self.revin(forecast, "denorm")

    def hybrid_loss(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Eq. (18): ``lambda MSE + (1 - lambda) MAE``."""
        return self.mse_weight * F.mse_loss(forecast, target) + (1 - self.mse_weight) * F.l1_loss(forecast, target)

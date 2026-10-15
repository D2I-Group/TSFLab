"""WDAN: wavelet-based disentangled adaptive normalization around a PatchTST backbone.

Independent implementation from the Methodology section (Eqs. 1-17) of Lin et al.,
"Wavelet-based Disentangled Adaptive Normalization for Non-stationary Times Series
Forecasting" (arXiv 2506.05857), after reading the pinned official code
(``MonBG/WDAN`` at ``f01994ad``, no license file) to resolve omissions; nothing is
copied or imported. The backbone is the catalog ``patchtst`` component with its own
instance normalization disabled, as in the official runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.patchtst import PatchTSTBackbone
from tsflab.models._inputs import check_window

_EPS = 1e-5

# Orthonormal decomposition low-pass filters (standard tabulated constants). The
# high-pass filter is the quadrature mirror ``g[k] = (-1)^(k+1) h[L-1-k]``.
_LOW_PASS: dict[str, tuple[float, ...]] = {
    "haar": (2**-0.5, 2**-0.5),
    "db4": (
        -0.010597401785069032,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ),
    "coif3": (
        -3.459977319727278e-05,
        -7.0983302506379e-05,
        0.0004662169598204029,
        0.0011175187708306303,
        -0.0025745176881367972,
        -0.009007976136730624,
        0.015880544863669452,
        0.03455502757329774,
        -0.08230192710629983,
        -0.07179982161915484,
        0.42848347637737,
        0.7937772226260872,
        0.40517690240911824,
        -0.06112339000297255,
        -0.06577191128146936,
        0.023452696142077168,
        0.007782596425672746,
        -0.003793512864380802,
    ),
}


def wavelet_filters(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Decomposition low-pass and quadrature-mirror high-pass filters."""
    low = torch.tensor(_LOW_PASS[name], dtype=torch.float32)
    signs = torch.tensor([(-1.0) ** (k + 1) for k in range(low.numel())])
    return low, signs * low.flip(0)


def _right_extension(length: int, taps: int) -> int:
    return taps - 1 if (length + taps) % 2 == 0 else taps


def dwt_level_lengths(length: int, taps: int, levels: int) -> list[int]:
    """Signal length entering each analysis level (symmetric extension, stride 2)."""
    lengths = []
    for _ in range(levels):
        if length < taps:
            raise ValueError(
                f"a length-{length} signal is shorter than the {taps}-tap wavelet filter; "
                "use fewer DWT levels, a shorter wavelet, or longer windows"
            )
        lengths.append(length)
        length = (length + _right_extension(length, taps) - 1) // 2 + 1
    return lengths


def symmetric_extend(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Half-sample symmetric extension along the last axis (edge samples repeated)."""
    parts = [x[..., :left].flip(-1), x] if left else [x]
    if right:
        parts.append(x[..., x.shape[-1] - right :].flip(-1))
    return torch.cat(parts, dim=-1)


class WaveletTrend(nn.Module):
    """Eqs. (1)-(3): K-level DWT with fixed orthonormal filters; the trend ``x_l`` is
    the inverse transform of the level-K approximation with every detail band zeroed.

    Analysis correlates the symmetrically extended signal with the decomposition
    filters at stride 2; synthesis is the adjoint (transposed) operation cropped to
    the input alignment, so synthesizing all bands of an analysis returns the input
    for an orthonormal filter pair.
    """

    def __init__(self, wavelet: str, levels: int) -> None:
        super().__init__()
        low, high = wavelet_filters(wavelet)
        self.register_buffer("low", low.view(1, 1, -1), persistent=False)
        self.register_buffer("high", high.view(1, 1, -1), persistent=False)
        self.taps = low.numel()
        self.levels = levels

    def analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One level on ``[M, 1, T]``: approximation and detail coefficients."""
        padded = symmetric_extend(x, self.taps - 1, _right_extension(x.shape[-1], self.taps))
        return F.conv1d(padded, self.low, stride=2), F.conv1d(padded, self.high, stride=2)

    def synthesis(self, approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        """Adjoint of :meth:`analysis` (one level, cropped to the input grid)."""
        pad = self.taps - 1
        return F.conv_transpose1d(approx, self.low, stride=2, padding=pad) + F.conv_transpose1d(
            detail, self.high, stride=2, padding=pad
        )

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``[M, 1, T]`` -> level-K approximation and details ``[c_h1, ..., c_hK]``."""
        details = []
        approx = x
        for _ in range(self.levels):
            approx, detail = self.analysis(approx)
            details.append(detail)
        return approx, details

    def reconstruct(self, approx: torch.Tensor, details: list[torch.Tensor], length: int) -> torch.Tensor:
        """Inverse of :meth:`decompose`, trimming the one-sample overhang of odd levels."""
        current = approx
        for detail in reversed(details):
            if current.shape[-1] > detail.shape[-1]:
                current = current[..., : detail.shape[-1]]
            current = self.synthesis(current, detail)
        return current[..., :length]

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """Low-frequency trend of ``series`` ``[B, T, C]`` (Eq. 3, ``x_l``), same shape."""
        batch, length, channels = series.shape
        flat = series.transpose(1, 2).reshape(batch * channels, 1, length)
        approx, details = self.decompose(flat)
        trend = self.reconstruct(approx, [torch.zeros_like(d) for d in details], length)
        return trend.reshape(batch, channels, length).transpose(1, 2)


def sliding_std(series: torch.Tensor, window: int) -> torch.Tensor:
    """Eqs. (4)-(5): point-level standard deviation over a sliding window along time
    (population form), replication-padded back to the input length; ``[B, T, C]``.

    For an even window the extra padded sample sits on the left, as in the official code.
    """
    length = series.shape[1]
    if window < 1 or window > length:
        raise ValueError(f"window_len={window} must lie in [1, {length}]")
    windows = series.transpose(1, 2).unfold(-1, window, 1)  # [B, C, T - w + 1, w]
    std = windows.var(dim=-1, unbiased=False).sqrt()
    left = window // 2
    if window > 1:
        std = F.pad(std, (left, window - 1 - left), mode="replicate")
    return std.transpose(1, 2)


class FeedForward(nn.Module):
    """Bias-free two-layer GELU perceptron with dropout (no residual)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatisticsPredictor(nn.Module):
    """Eqs. (7)-(12): point-level future mean and standard deviation series.

    ``MLP_1..MLP_4`` are single linear maps along time (with dropout) for the centered
    mean series, its first difference, the high-frequency residual, and the centered
    standard deviation series; ``MLP_mu``/``MLP_sigma`` fuse their concatenations
    (a linear fusion, ``ffn_layers`` feed-forward blocks, and a linear head to the
    horizon). ``MLP_1`` and ``MLP_3`` are shared by both branches.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int, d_ff: int, dropout: float, ffn_layers: int) -> None:
        super().__init__()

        def embed(length: int) -> nn.Module:
            return nn.Sequential(nn.Linear(length, d_model), nn.Dropout(dropout))

        self.mean_embed = embed(seq_len)  # MLP_1
        self.diff_embed = embed(seq_len - 1)  # MLP_2
        self.residual_embed = embed(seq_len)  # MLP_3
        self.std_embed = embed(seq_len)  # MLP_4
        self.mean_fuse = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Dropout(dropout))
        self.std_fuse = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Dropout(dropout))
        self.mean_ffn = nn.Sequential(*[FeedForward(d_model, d_ff, dropout) for _ in range(ffn_layers)])
        self.std_ffn = nn.Sequential(*[FeedForward(d_model, d_ff, dropout) for _ in range(ffn_layers)])
        self.mean_head = nn.Linear(d_model, pred_len)
        self.std_head = nn.Linear(d_model, pred_len)

    def forward(
        self, mean: torch.Tensor, std: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Inputs [B, C, T] (time last); outputs [B, C, H].
        mean_bar = mean.mean(dim=-1, keepdim=True)  # Eq. (7)
        std_bar = std.mean(dim=-1, keepdim=True)
        centered_mean = mean - mean_bar  # Eq. (8)
        centered_std = std - std_bar  # Eq. (9)
        mean_diff = centered_mean[..., 1:] - centered_mean[..., :-1]  # Eq. (10)
        mean_feature = self.mean_embed(centered_mean)
        residual_feature = self.residual_embed(residual)
        fused_mean = self.mean_fuse(torch.cat([mean_feature, self.diff_embed(mean_diff), residual_feature], dim=-1))
        fused_std = self.std_fuse(torch.cat([self.std_embed(centered_std), mean_feature, residual_feature], dim=-1))
        future_mean = self.mean_head(self.mean_ffn(fused_mean)) + mean_bar  # Eq. (11)
        future_std = F.relu(self.std_head(self.std_ffn(fused_std)) + std_bar)  # Eq. (12), kept >= 0
        return future_mean, future_std


class Model(nn.Module):
    """Wavelet-disentangled point-level normalization, PatchTST backbone, predicted
    point-level statistics for de-normalization, three-stage training."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        wavelet: str = "coif3",
        dwt_levels: int = 1,
        window_len: int = 5,
        stat_d_model: int = 512,
        stat_d_ff: int = 1024,
        stat_dropout: float = 0.1,
        stat_ffn_layers: int = 2,
        stat_pretrain_epochs: int = 5,
        stat_lr: float = 1e-4,
        joint_after_epochs: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        e_layers: int = 3,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if not 1 <= window_len <= min(seq_len, pred_len):
            raise ValueError("window_len must lie in [1, min(seq_len, pred_len)]")
        self.trend = WaveletTrend(wavelet, dwt_levels)
        # Both the input window and the horizon (statistics targets) are decomposed.
        dwt_level_lengths(seq_len, self.trend.taps, dwt_levels)
        dwt_level_lengths(pred_len, self.trend.taps, dwt_levels)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.window_len = window_len
        self.stat_pretrain_epochs = stat_pretrain_epochs
        self.stat_lr = stat_lr
        self.joint_after_epochs = joint_after_epochs
        self.statistics = StatisticsPredictor(seq_len, pred_len, stat_d_model, stat_d_ff, stat_dropout, stat_ffn_layers)
        self.backbone = PatchTSTBackbone(
            c_in=enc_in,
            context_window=seq_len,
            target_window=pred_len,
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            n_layers=e_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=None,
            d_v=None,
            d_ff=d_ff,
            activation="gelu",
            norm="BatchNorm",
            attn_dropout=0.0,
            res_dropout=dropout,
            ffn_dropout=dropout,
            proj_dropout=0.0,
            head_dropout=0.0,
            pre_norm=False,
            pe="zeros",
            learn_pe=False,
            head_type="flatten",
            individual=False,
            revin=False,
            affine=False,
            subtract_last=False,
        )
        # Stage bookkeeping saved with checkpoints: training batches per epoch and
        # catalog-stage batches seen so far.
        self.register_buffer("steps_per_epoch", torch.ones((), dtype=torch.long))
        self.register_buffer("trained_batches", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------- Eqs. 1-14
    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trend ``x_l`` (the mean series), residual ``x_h = x - x_l``, and the sliding
        standard deviation of ``x_h``; all ``[B, T, C]``."""
        trend = self.trend(x)
        residual = x - trend
        return trend, residual, sliding_std(residual, self.window_len)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Eq. (6) plus the statistics prediction of Eqs. (7)-(12).

        Returns the stationary window ``[B, L, C]`` and the predicted future mean and
        standard deviation series, each ``[B, H, C]``.
        """
        trend, residual, std = self.decompose(x)
        normalized = (x - trend) / (std + _EPS)
        mean, scale = self.statistics(trend.transpose(1, 2), std.transpose(1, 2), residual.transpose(1, 2))
        return normalized, (mean.transpose(1, 2), scale.transpose(1, 2))

    @staticmethod
    def denormalize(output: torch.Tensor, statistics: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Eq. (14): ``y = y_bar * (sigma_hat + eps) + mu_hat``."""
        mean, std = statistics
        return output * (std + _EPS) + mean

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized, statistics = self.normalize(x_enc)
        return self.denormalize(self.backbone(normalized), statistics)  # Eqs. (13)-(14)

    # ------------------------------------------------------------ Eqs. 15-17
    def statistics_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Ground-truth future statistics: the horizon's trend and the sliding standard
        deviation of its residual, stacked as ``[B, 2, H, C]``."""
        trend, _, std = self.decompose(y)
        return torch.stack([trend, std], dim=1)

    def statistics_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Eq. (15) with MSE between predicted and true future statistics."""
        _, (mean, std) = self.normalize(x)
        return F.mse_loss(torch.stack([mean, std], dim=1), self.statistics_targets(y))

    def set_statistics_trainable(self, trainable: bool) -> None:
        for parameter in self.statistics.parameters():
            parameter.requires_grad_(trainable)

    def pretrain(self, train_loader, device) -> None:
        """Stage 1 (Eq. 15): fit the statistics predictor alone on the training split,
        then freeze it for stage 2."""
        self.statistics.to(device).train()
        self.set_statistics_trainable(True)
        optimizer = torch.optim.Adam(self.statistics.parameters(), lr=self.stat_lr)
        for _ in range(self.stat_pretrain_epochs):
            for batch in train_loader:
                x = batch[0].float().to(device)
                y = batch[1].float().to(device)[:, -self.pred_len :, : self.enc_in]
                optimizer.zero_grad()
                self.statistics_loss(x, y).backward()
                optimizer.step()
        self.set_statistics_trainable(False)

    def prepare_stages(self, steps_per_epoch: int) -> None:
        """Record the training batches per epoch before fresh catalog training."""
        self.steps_per_epoch.fill_(max(int(steps_per_epoch), 1))
        self.trained_batches.zero_()

    def advance_stage(self) -> bool:
        """Stage 2 (Eq. 16, statistics frozen) for the first ``joint_after_epochs``
        epochs of catalog training, stage 3 (Eq. 17, joint) afterwards. Called once per
        training batch; returns whether the joint stage is active."""
        joint = int(self.trained_batches) >= self.joint_after_epochs * int(self.steps_per_epoch)
        self.set_statistics_trainable(joint)
        self.trained_batches += 1
        return joint

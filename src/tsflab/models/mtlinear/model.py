"""Paper-driven local implementation of MTLinear (multi-task linear forecasting).

Variates are grouped a priori by correlation; each group owns one linear-family
forecaster that is shared by its variates (Sec. 4.2, 4.4), and training weights the
squared error of every (horizon step, variate) cell by a detached inverse error
magnitude (Sec. 4.3, Eq. 7-8).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from torch import nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._components.last_value_center import (
    center_on_last_value,
    restore_last_value,
)
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window

LAYER_TYPES = ("DLinear", "NLinear", "Linear", "RLinear")


class GroupHead(nn.Module):
    """One univariate linear-family forecaster applied to ``(N, L, 1)`` series."""

    def __init__(self, layer_type: str, seq_len: int, pred_len: int, kernel_size: int) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "DLinear":
            self.net = DLinearBackbone(1, seq_len, pred_len, kernel_size=kernel_size)
        else:
            self.net = ChannelWiseLinear(seq_len, pred_len, 1)
        if layer_type == "RLinear":
            self.norm = RevIN(1, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "DLinear":
            return self.net(x)
        if self.layer_type == "NLinear":
            centered, level = center_on_last_value(x)
            return restore_last_value(self.net(centered.transpose(1, 2)).transpose(1, 2), level)
        if self.layer_type == "RLinear":
            x = self.norm(x, "norm")
            return self.norm(self.net(x.transpose(1, 2)).transpose(1, 2), "denorm")
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    """Route each variate group to its own linear forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        layer_type: str = "DLinear",
        kernel_size: int = 25,
        cluster_dist: float = 0.5,
        max_clusters: int = 8,
        penalty_param: float = 2.0,
        use_horizon_penalty: bool = True,
        use_variates_penalty: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.cluster_dist = float(cluster_dist)
        self.num_groups = min(max_clusters, enc_in)
        self.penalty_param = float(penalty_param)
        self.use_horizon_penalty = bool(use_horizon_penalty)
        self.use_variates_penalty = bool(use_variates_penalty)
        self.heads = nn.ModuleList(
            GroupHead(layer_type, seq_len, pred_len, kernel_size) for _ in range(self.num_groups)
        )
        # Before ``fit_clusters`` runs the variates are spread round-robin; the
        # trainer fits the data-driven grouping once on the training split.
        self.register_buffer("group_of", torch.arange(enc_in) % self.num_groups)
        self.register_buffer("clusters_fitted", torch.tensor(False))

    # -- Sec. 4.2: a-priori correlation clustering ---------------------------------
    @torch.no_grad()
    def fit_clusters(self, series: torch.Tensor) -> torch.Tensor:
        """Complete-linkage clustering of variates on ``1 - corr`` of ``(T, C)`` data."""
        if series.ndim != 2 or series.shape[1] != self.enc_in or series.shape[0] < 2:
            raise ValueError("series must have shape (T >= 2, enc_in)")
        corr = np.nan_to_num(np.corrcoef(series.double().cpu().numpy().T), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        if self.enc_in == 1:
            labels = np.zeros(1, dtype=np.int64)
        else:
            dist = np.clip(1.0 - corr, 0.0, 2.0)
            tree = linkage(squareform(dist, checks=False), method="complete")
            # merges happen only strictly below the threshold
            labels = fcluster(tree, t=np.nextafter(self.cluster_dist, -1.0), criterion="distance")
            if labels.max() > self.num_groups:
                labels = fcluster(tree, t=self.num_groups, criterion="maxclust")
        _, first = np.unique(labels, return_index=True)
        order = {int(labels[i]): rank for rank, i in enumerate(np.sort(first))}
        ids = torch.tensor([order[int(v)] for v in labels], dtype=torch.long)
        self.group_of.copy_(ids.to(self.group_of.device))
        self.clusters_fitted.fill_(True)
        return self.group_of

    # -- Sec. 4.4: one linear forecaster per group ---------------------------------
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        out = x_enc.new_zeros(batch, self.pred_len, self.enc_in)
        for group, head in enumerate(self.heads):
            index = (self.group_of == group).nonzero(as_tuple=True)[0]
            if index.numel() == 0:
                continue
            series = x_enc.index_select(2, index).permute(0, 2, 1).reshape(-1, self.seq_len, 1)
            forecast = head(series).reshape(batch, index.numel(), self.pred_len).permute(0, 2, 1)
            out = out.index_copy(2, index, forecast)
        return out

    # -- Sec. 4.3, Eq. (7)-(8): gradient magnitude penalty -----------------------------
    @torch.no_grad()
    def penalty_weights(self, error: torch.Tensor) -> torch.Tensor:
        """Constant ``(H, C)`` weights from absolute errors ``(B, H, C)``, per group."""
        horizon, channels = error.shape[1:]
        groups = self.group_of[self.enc_in - channels :]
        weights = error.new_zeros(horizon, channels)

        def inverse(value: torch.Tensor) -> torch.Tensor:
            return torch.where(value == 0, torch.ones_like(value), value).pow(-self.penalty_param)

        for group in groups.unique():
            cols = (groups == group).nonzero(as_tuple=True)[0]
            cell = error[:, :, cols]
            by_variate = inverse(cell.mean(dim=(0, 1)))  # 1 / H_i^a
            by_horizon = inverse(cell.mean(dim=(0, 2)))  # 1 / K_j^a
            if self.use_horizon_penalty and self.use_variates_penalty:
                weights[:, cols] = by_horizon[:, None] * by_variate[None, :]
            elif self.use_variates_penalty:
                weights[:, cols] = by_variate[None, :]
            else:
                weights[:, cols] = by_horizon[:, None] * (cols.numel() / channels)
        return weights

    def penalized_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted squared error ``sum_ij w_ij (pred_ij - y_ij)^2`` averaged over the batch."""
        squared = (prediction - target).pow(2)
        if not (self.use_horizon_penalty or self.use_variates_penalty):
            return squared.mean()
        weights = self.penalty_weights((prediction - target).abs().detach())
        cell = squared.mean(dim=0)  # (H, C)
        if not self.use_horizon_penalty:
            return (weights[0] * cell.mean(dim=0)).sum()
        return (weights * cell).sum()

"""Local APTF: amortized hierarchical predictability-aware training around PatchTST.

Paper: Amortized Predictability-aware Training Framework for Time Series
Forecasting and Classification (WWW 2026), Section 3, Algorithms 1-2.

APTF changes only how a forecaster is trained. Each batch is split into buckets
by ascending per-sample loss; low-predictability (high-loss) buckets receive
smaller loss weights (Algorithm 1). Training is divided into stages of
``stage_epochs`` epochs; stage ``S`` averages ``S`` bucket groups, group ``i``
dropping the ``i`` largest weights (hierarchical predictability-aware loss).
An amortization model of the same architecture estimates the buckets used for
the source model and vice versa (Algorithm 2). Inference uses the source model.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from tsflab.models._components.patchtst import PatchTSTBackbone


def bucket_weights(num_buckets: int) -> list[float]:
    """First-stage weights: 1 decreasing by ``1 / (K - 1)``; the last is half the previous."""
    step = 1.0 / (num_buckets - 1)
    weights = [1.0 - index * step for index in range(num_buckets)]
    weights[-1] = weights[-2] / 2.0
    return weights


def split_buckets(order: torch.Tensor, count: int) -> list[torch.Tensor]:
    """Split ascending-loss sample indices into ``count`` buckets.

    The first ``count - 1`` buckets hold ``floor(N / count)`` samples each and the
    last bucket takes the remainder; buckets may be empty for small batches.
    """
    size = order.numel() // count
    buckets = [order[index * size : (index + 1) * size] for index in range(count - 1)]
    buckets.append(order[(count - 1) * size :])
    return buckets


def predictability_aware_loss(
    outputs: torch.Tensor,
    target: torch.Tensor,
    order: torch.Tensor,
    weights: list[float],
    criterion: Callable,
) -> torch.Tensor:
    """Algorithm 1: sum over buckets of ``weight_j * criterion(bucket_j)``.

    ``order`` ranks samples by ascending loss (most predictable first), so bucket
    ``j`` with a larger index holds less predictable samples and a smaller weight.
    """
    loss = outputs.new_zeros(())
    for bucket, weight in zip(split_buckets(order, len(weights)), weights):
        if bucket.numel():
            loss = loss + weight * criterion(outputs[bucket], target[bucket])
    return loss


class Model(nn.Module):
    """PatchTST source forecaster plus an optional amortization copy for APTF training."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_buckets: int = 9,
        stage_epochs: int = 2,
        amortization: bool = True,
        patch_len: int = 16,
        stride: int = 8,
        e_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.0,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.num_buckets = num_buckets
        self.stage_epochs = stage_epochs
        self.amortization = amortization
        self.weights = bucket_weights(num_buckets)
        backbone = dict(
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
            head_dropout=head_dropout,
            pre_norm=False,
            pe="zeros",
            learn_pe=True,
            head_type="flatten",
            individual=False,
            revin=revin,
            affine=False,
            subtract_last=False,
        )
        self.source = PatchTSTBackbone(**backbone)
        # Independently initialised model of the same architecture (Section 3.2).
        self.amortizer = PatchTSTBackbone(**backbone) if amortization else None
        # Training progress, used to derive the epoch and hence the stage.
        self.register_buffer("steps_per_epoch", torch.zeros((), dtype=torch.long))
        self.register_buffer("train_step", torch.zeros((), dtype=torch.long))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        return self.source(x_enc)

    @property
    def num_groups(self) -> int:
        """Total number of stages ``G``: bucket counts ``K, K-1, ..., 1``."""
        return self.num_buckets

    def stage(self, epoch: int) -> int:
        """Zero-based stage: advances every ``stage_epochs`` epochs (Eq. 2), capped at ``G - 1``."""
        return min(epoch // self.stage_epochs, self.num_groups - 1)

    def current_epoch(self) -> int:
        steps = int(self.steps_per_epoch)
        return int(self.train_step) // steps if steps > 0 else 0

    def hierarchical_loss(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        order: torch.Tensor,
        stage: int,
        criterion: Callable,
    ) -> torch.Tensor:
        """Hierarchical predictability-aware loss at ``stage``, divided by ``G``.

        Group ``i`` (``0 <= i <= stage``) uses ``K - i`` buckets with the weights
        that remain after removing the ``i`` largest first-stage weights.
        """
        total = outputs.new_zeros(())
        for group in range(stage + 1):
            total = total + predictability_aware_loss(outputs, target, order, self.weights[group:], criterion)
        return total / self.num_groups


def loss_order(outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sample indices sorted by ascending per-sample mean squared error (detached)."""
    errors = (outputs.detach() - target).pow(2).flatten(1).mean(dim=1)
    return torch.argsort(errors)

"""Built-in loss implementations and registration."""

from __future__ import annotations

import torch.nn as nn

from tsflab.catalog.registry import LOSS_REGISTRY


def register() -> None:
    """Register built-in losses into the registry."""
    LOSS_REGISTRY.register("mse", lambda **kwargs: nn.MSELoss(**kwargs))
    LOSS_REGISTRY.register("mae", lambda **kwargs: nn.L1Loss(**kwargs))
    LOSS_REGISTRY.register("huber", lambda **kwargs: nn.HuberLoss(**kwargs))

"""Independent differentiable extremely-randomized-tree baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.soft_tree import SoftDecisionTree
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Average trees with frozen random axis-aligned splits and learned leaves."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_estimators: int = 24,
        tree_depth: int = 2,
        threshold_range: float = 1.0,
        temperature: float = 1.0,
        random_seed: int = 1733,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        input_dim, output_dim = seq_len * enc_in, pred_len * enc_in
        generator = torch.Generator().manual_seed(random_seed)
        split_count = 2**tree_depth - 1
        trees = []
        for _ in range(num_estimators):
            feature = torch.randint(input_dim, (split_count,), generator=generator)
            weight = torch.zeros(split_count, input_dim)
            weight[torch.arange(split_count), feature] = 1.0
            threshold = torch.empty(split_count).uniform_(-threshold_range, threshold_range, generator=generator)
            trees.append(
                SoftDecisionTree(
                    input_dim, output_dim, tree_depth, temperature, fixed_split_weight=weight, fixed_threshold=threshold
                )
            )
        self.trees = nn.ModuleList(trees)
        self.revin = RevIN(enc_in, affine=use_revin, enabled=use_revin)
        self.aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm").flatten(1)
        forecast = torch.stack([tree(normalized) for tree in self.trees]).mean(dim=0)
        forecast = forecast.view(-1, self.pred_len, self.enc_in)
        self.aux_loss = forecast.new_zeros(())
        return self.revin(forecast, "denorm")

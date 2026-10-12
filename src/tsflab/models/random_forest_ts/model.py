"""Independent differentiable randomized-tree averaging baseline."""

from __future__ import annotations
import torch
import torch.nn as nn
from tsflab.models._components.revin import RevIN
from tsflab.models._components.soft_tree import SoftDecisionTree

class Model(nn.Module):
    """Average soft trees fitted through fixed random feature subspaces."""
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, num_estimators: int = 16,
                 tree_depth: int = 3, feature_fraction: float = 0.7,
                 temperature: float = 1.0, random_seed: int = 1729,
                 use_revin: bool = True) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("dimensions and num_estimators must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        input_dim, output_dim = seq_len * enc_in, pred_len * enc_in
        generator = torch.Generator().manual_seed(random_seed)
        split_count = 2**tree_depth - 1
        trees = []
        for _ in range(num_estimators):
            mask = torch.rand(split_count, input_dim, generator=generator) < feature_fraction
            empty = ~mask.any(dim=-1)
            if empty.any():
                chosen = torch.randint(input_dim, (int(empty.sum()),), generator=generator)
                mask[empty, chosen] = True
            trees.append(SoftDecisionTree(input_dim, output_dim, tree_depth, temperature,
                                          split_mask=mask))
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
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        normalized = self.revin(x_enc, "norm").flatten(1)
        forecast = torch.stack([tree(normalized) for tree in self.trees]).mean(dim=0)
        forecast = forecast.view(-1, self.pred_len, self.enc_in)
        self.aux_loss = forecast.new_zeros(())
        return self.revin(forecast, "denorm")

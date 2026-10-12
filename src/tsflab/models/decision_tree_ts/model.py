"""Independent differentiable single-tree forecasting baseline."""

from __future__ import annotations
import torch
import torch.nn as nn
from tsflab.models._components.revin import RevIN
from tsflab.models._components.soft_tree import SoftDecisionTree

class Model(nn.Module):
    """Map a flattened lag window through one soft binary regression tree."""
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, tree_depth: int = 4,
                 temperature: float = 1.0, use_revin: bool = True) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in, affine=use_revin, enabled=use_revin)
        self.tree = SoftDecisionTree(seq_len * enc_in, pred_len * enc_in,
                                     depth=tree_depth, temperature=temperature)
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
        normalized = self.revin(x_enc, "norm")
        forecast = self.tree(normalized.flatten(1)).view(-1, self.pred_len, self.enc_in)
        self.aux_loss = forecast.new_zeros(())
        return self.revin(forecast, "denorm")

"""Independent differentiable RBF-basis epsilon-regression baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Forecast with learned RBF support centres and epsilon-insensitive loss."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, num_support: int = 16, kernel_gamma: float = 0.1, epsilon: float = 0.1, l2_penalty: float = 1e-4) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("dimensions and num_support must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.kernel_gamma, self.epsilon, self.l2_penalty = kernel_gamma, epsilon, l2_penalty
        self.support_centres = nn.Parameter(torch.randn(num_support, seq_len) * 0.05)
        self.coefficients = nn.Parameter(torch.randn(num_support, pred_len) * 0.05)
        self.bias = nn.Parameter(torch.zeros(pred_len))
        self.aux_loss: torch.Tensor | None = None

    def epsilon_insensitive_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        return (prediction.sub(target).abs() - self.epsilon).clamp_min(0).mean()

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        queries = x_enc.transpose(1, 2).reshape(-1, self.seq_len)
        features = torch.exp(-self.kernel_gamma * torch.cdist(queries, self.support_centres).square())
        forecast = features @ self.coefficients + self.bias
        self.aux_loss = 0.5 * self.l2_penalty * self.coefficients.square().sum()
        return forecast.reshape(x_enc.shape[0], self.enc_in, self.pred_len).transpose(1, 2)

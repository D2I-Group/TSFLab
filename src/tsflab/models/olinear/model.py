"""Independent OLinear implementation from the published method equations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class NormLin(nn.Module):
    """Positive row-normalized channel mixing from OLinear equation (3)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.normal_(self.weight, std=0.02)

    def normalized_weight(self) -> torch.Tensor:
        weight = F.softplus(self.weight)
        return weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oc,bcld->bold", self.normalized_weight(), x)


class Model(nn.Module):
    """Compact OLinear whose transform bases are explicit serialized buffers."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        dropout: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_revin = use_revin
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.register_buffer("input_basis", torch.eye(seq_len), persistent=True)
        self.register_buffer("output_basis", torch.eye(pred_len), persistent=True)
        self.embedding = nn.Linear(1, d_model)
        self.channel_in = nn.Linear(d_model, d_model)
        self.channel_mix = NormLin(enc_in)
        self.channel_out = nn.Linear(d_model, d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.sequence_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.sequence_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(seq_len * d_model, pred_len)

    @torch.no_grad()
    def set_orthogonal_bases(
        self, input_basis: torch.Tensor, output_basis: torch.Tensor
    ) -> None:
        """Install Pearson-correlation eigenvector bases computed on training data."""
        if input_basis.shape != (self.seq_len, self.seq_len):
            raise ValueError("input basis has the wrong shape")
        if output_basis.shape != (self.pred_len, self.pred_len):
            raise ValueError("output basis has the wrong shape")
        eye_in = torch.eye(self.seq_len, device=input_basis.device, dtype=input_basis.dtype)
        eye_out = torch.eye(self.pred_len, device=output_basis.device, dtype=output_basis.dtype)
        torch.testing.assert_close(input_basis.T @ input_basis, eye_in, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(output_basis.T @ output_basis, eye_out, atol=1e-4, rtol=1e-4)
        self.input_basis.copy_(input_basis)
        self.output_basis.copy_(output_basis)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        if self.use_revin:
            x_enc = self.revin(x_enc, "norm")
        transformed = torch.einsum("lt,btc->blc", self.input_basis.T, x_enc)
        hidden = self.embedding(transformed.permute(0, 2, 1).unsqueeze(-1))
        channel_update = self.channel_out(self.channel_mix(self.channel_in(hidden)))
        hidden = self.channel_norm(hidden + channel_update)
        hidden = self.sequence_norm(hidden + self.sequence_mlp(hidden))
        forecast_domain = self.decoder(hidden.flatten(start_dim=2))
        output = torch.einsum("ht,bct->bhc", self.output_basis, forecast_domain)
        if self.use_revin:
            output = self.revin(output, "denorm")
        return output

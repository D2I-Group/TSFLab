"""Clean-room SEMPO forecaster from EASD and MoPFormer equations.

The runtime model uses differentiable deterministic spectral masks in place of
the paper's stochastic pre-training masks. It includes patch tokens, routed
prompt key/value tokens, and a prediction head, but no released pre-trained
weights, reconstruction corpus, or two-stage training harness.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 patch_len: int = 16, num_prompts: int = 4, num_heads: int = 4,
                 dropout: float = 0.1, use_revin: bool = True) -> None:
        super().__init__()
        if seq_len % patch_len:
            raise ValueError("seq_len must be divisible by patch_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.num_patches, self.use_revin = patch_len, seq_len // patch_len, use_revin
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.energy_threshold = nn.Parameter(torch.tensor(0.0))
        self.high_mask_logits = nn.Parameter(torch.full((seq_len // 2 + 1,), 0.2))
        self.low_mask_logits = nn.Parameter(torch.full((seq_len // 2 + 1,), -0.2))
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(self.num_patches, d_model))
        self.prompt_experts = nn.Parameter(torch.empty(num_prompts, d_model))
        self.router = nn.Linear(d_model, num_prompts)
        self.prompt_kv = nn.Linear(d_model, 2 * d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.prompt_experts, std=0.02)

    def energy_aware_decomposition(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(x, dim=-1)
        energy = spectrum.abs().square()
        threshold = energy.mean(dim=-1, keepdim=True) * self.energy_threshold.exp()
        high_selector = torch.sigmoid((energy - threshold) / threshold.clamp_min(1e-6))
        high = spectrum * high_selector * torch.sigmoid(self.high_mask_logits)
        low = spectrum * (1 - high_selector) * torch.sigmoid(self.low_mask_logits)
        reconstructed = torch.fft.irfft(high + low, n=self.seq_len, dim=-1)
        return reconstructed, high, low

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        normalized = self.revin(x_enc, "norm") if self.use_revin else x_enc
        channel_history = normalized.transpose(1, 2)
        masked, _, _ = self.energy_aware_decomposition(channel_history)
        tokens = masked.unfold(-1, self.patch_len, self.patch_len)
        tokens = self.patch_projection(tokens) + self.position
        batch, channels, patches, width = tokens.shape
        tokens = tokens.reshape(batch * channels, patches, width)
        routing = self.router(tokens).softmax(-1)
        mixed = routing @ self.prompt_experts
        prompt_key, prompt_value = self.prompt_kv(mixed).chunk(2, dim=-1)
        attended, _ = self.attention(tokens, torch.cat((prompt_key, tokens), 1), torch.cat((prompt_value, tokens), 1), need_weights=False)
        hidden = self.norm1(tokens + attended)
        hidden = self.norm2(hidden + self.feed_forward(hidden))
        forecast = self.head(hidden.flatten(1)).reshape(batch, channels, self.pred_len).transpose(1, 2)
        return self.revin(forecast, "denorm") if self.use_revin else forecast

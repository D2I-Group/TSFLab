"""Clean-room MMPD loss with a patch-consistent conditional denoiser."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNMLPBlock(nn.Module):
    """Appendix-B AdaLN-MLP recurrence from Eq. (12)."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.modulation = nn.Linear(dimension, 3 * dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gate, scale, shift = self.modulation(F.silu(condition)).chunk(3, -1)
        adapted = (1 + scale) * self.norm(x) + shift
        return x + gate * self.mlp(adapted)


class PatchConsistentDenoiser(nn.Module):
    """Equation (7): token, step, previous, and next patch conditioning."""

    def __init__(
        self,
        patch_len: int,
        dimension: int,
        adjacent_range: int,
        diffusion_steps: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.adjacent_range = adjacent_range
        self.patch_projection = nn.Linear(patch_len, dimension)
        self.token_projection = nn.Linear(dimension, dimension)
        self.step_embedding = nn.Embedding(diffusion_steps, dimension)
        neighbour_width = max(1, adjacent_range * patch_len)
        self.previous_projection = nn.Linear(neighbour_width, dimension)
        self.next_projection = nn.Linear(neighbour_width, dimension)
        self.blocks = nn.ModuleList(
            [AdaLNMLPBlock(dimension, dropout) for _ in range(depth)]
        )
        self.final_modulation = nn.Linear(dimension, 2 * dimension)
        self.output = nn.Linear(dimension, patch_len)

    def adjacent_patches(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, width = patches.shape
        if self.adjacent_range == 0:
            zeros = patches.new_zeros(batch, count, 1)
            return zeros, zeros
        padded = F.pad(patches.transpose(1, 2), (self.adjacent_range, self.adjacent_range))
        windows = padded.unfold(-1, 2 * self.adjacent_range + 1, 1).permute(0, 2, 1, 3)
        previous = windows[..., : self.adjacent_range].reshape(batch, count, -1)
        following = windows[..., self.adjacent_range + 1 :].reshape(batch, count, -1)
        return previous, following

    def forward(
        self, noisy_patches: torch.Tensor, tokens: torch.Tensor, step: int | torch.Tensor
    ) -> torch.Tensor:
        if noisy_patches.ndim != 3 or noisy_patches.shape[-1] != self.patch_len:
            raise ValueError("noisy patches have the wrong shape")
        if tokens.shape[:2] != noisy_patches.shape[:2]:
            raise ValueError("one future token is required for every noisy patch")
        batch, count, _ = noisy_patches.shape
        step_tensor = torch.as_tensor(step, device=noisy_patches.device, dtype=torch.long)
        if step_tensor.ndim == 0:
            step_tensor = step_tensor.expand(batch)
        if step_tensor.shape != (batch,):
            raise ValueError("step must be scalar or one value per flattened series")
        previous, following = self.adjacent_patches(noisy_patches)
        condition = (
            self.token_projection(tokens)
            + self.step_embedding(step_tensor)[:, None]
            + self.previous_projection(previous)
            + self.next_projection(following)
        )
        hidden = self.patch_projection(noisy_patches)
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale, shift = self.final_modulation(F.silu(condition)).chunk(2, -1)
        hidden = (1 + scale) * F.layer_norm(hidden, (hidden.shape[-1],)) + shift
        return self.output(hidden)


class FuturePatchBackbone(nn.Module):
    """Patch history encoder producing one conditional token per future patch."""

    def __init__(
        self,
        seq_len: int,
        num_future_patches: int,
        patch_len: int,
        dimension: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.history_embedding = nn.Linear(patch_len, dimension)
        self.future_queries = nn.Parameter(torch.randn(num_future_patches, dimension) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )

    def history_patches(self, history: torch.Tensor) -> torch.Tensor:
        padding = (-history.shape[-1]) % self.patch_len
        if padding:
            history = F.pad(history, (padding, 0), mode="replicate")
        return history.unfold(-1, self.patch_len, self.patch_len)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        tokens = self.history_embedding(self.history_patches(history))
        queries = self.future_queries.unsqueeze(0).expand(history.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, tokens, tokens, need_weights=False)
        attended = self.norm(queries + attended)
        return self.norm(attended + self.feed_forward(attended))


class Model(nn.Module):
    """MMPD deterministic anchor path plus the paper's diffusion training loss."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        patch_len: int = 8,
        num_heads: int = 4,
        adjacent_range: int = 1,
        diffusion_steps: int = 100,
        denoiser_depth: int = 2,
        diffusion_weight: float = 0.99,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = min(patch_len, pred_len)
        self.num_future_patches = math.ceil(pred_len / self.patch_len)
        self.diffusion_weight = diffusion_weight
        self.backbone = FuturePatchBackbone(
            seq_len,
            self.num_future_patches,
            self.patch_len,
            d_model,
            num_heads,
            dropout,
        )
        self.denoiser = PatchConsistentDenoiser(
            self.patch_len,
            d_model,
            adjacent_range,
            diffusion_steps,
            denoiser_depth,
            dropout,
        )
        beta = torch.linspace(1e-4, 0.02, diffusion_steps)
        alpha_bar = torch.cumprod(1 - beta, 0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)
        self.anchor_step = int((alpha_bar - 0.5).abs().argmin())

    def _normalized_context(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )
        mean = x.mean(1, keepdim=True).detach()
        scale = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = (x - mean) / scale
        flattened = normalized.transpose(1, 2).reshape(x.shape[0] * self.enc_in, self.seq_len)
        tokens = self.backbone(flattened)
        return normalized, tokens, mean, scale

    def _target_patches(self, target: torch.Tensor) -> torch.Tensor:
        flattened = target.transpose(1, 2).reshape(target.shape[0] * self.enc_in, self.pred_len)
        total = self.num_future_patches * self.patch_len
        if total > self.pred_len:
            flattened = F.pad(flattened, (0, total - self.pred_len), mode="replicate")
        return flattened.reshape(-1, self.num_future_patches, self.patch_len)

    def diffusion_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Equation (8): random-step diffusion objective plus anchor forecast."""
        _, tokens, mean, scale = self._normalized_context(x)
        normalized_target = (target - mean) / scale
        clean = self._target_patches(normalized_target)
        step = torch.randint(len(self.alpha_bar), (), device=x.device)
        alpha = self.alpha_bar[step]
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        predicted_noise = self.denoiser(noisy, tokens, step)
        diffusion = F.mse_loss(predicted_noise, noise)

        anchor_alpha = self.alpha_bar[self.anchor_step]
        anchor_noise = self.denoiser(torch.zeros_like(clean), tokens, self.anchor_step)
        anchor_target = (anchor_alpha / (1 - anchor_alpha)).sqrt() * clean
        deterministic = F.mse_loss(anchor_noise, -anchor_target)
        return self.diffusion_weight * diffusion + (1 - self.diffusion_weight) * deterministic

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        _, tokens, mean, scale = self._normalized_context(x_enc)
        zeros = x_enc.new_zeros(
            x_enc.shape[0] * self.enc_in, self.num_future_patches, self.patch_len
        )
        predicted_noise = self.denoiser(zeros, tokens, self.anchor_step)
        alpha = self.alpha_bar[self.anchor_step]
        normalized = -((1 - alpha) / alpha).sqrt() * predicted_noise
        normalized = normalized.flatten(1)[:, : self.pred_len]
        forecast = normalized.reshape(x_enc.shape[0], self.enc_in, self.pred_len).transpose(1, 2)
        return forecast * scale + mean

    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_samples: int = 8, steps: int = 20) -> torch.Tensor:
        """Generate conditional diffusion samples; mode fitting remains outside point forecast."""
        if num_samples < 1 or steps < 1:
            raise ValueError("num_samples and steps must be positive")
        _, tokens, mean, scale = self._normalized_context(x)
        tokens = tokens.repeat(num_samples, 1, 1)
        noisy = torch.randn(
            tokens.shape[0], self.num_future_patches, self.patch_len,
            device=x.device, dtype=x.dtype,
        )
        schedule = torch.linspace(len(self.alpha_bar) - 1, 0, min(steps, len(self.alpha_bar))).long()
        for index, step in enumerate(schedule):
            alpha = self.alpha_bar[step]
            predicted_noise = self.denoiser(noisy, tokens, int(step))
            clean = (noisy - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if index + 1 < len(schedule):
                next_alpha = self.alpha_bar[schedule[index + 1]]
                noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * predicted_noise
            else:
                noisy = clean
        values = noisy.flatten(1)[:, : self.pred_len]
        values = values.reshape(num_samples, x.shape[0], self.enc_in, self.pred_len).permute(0, 1, 3, 2)
        return values * scale.unsqueeze(0) + mean.unsqueeze(0)

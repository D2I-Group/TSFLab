"""SiamTST: Siamese Time Series Transformer (arXiv 2407.02258).

A channel-independent patch Transformer (RevIN, non-overlapping patches, learnable
positional table) with pre-RMSNorm encoder layers, bias-free projections and
QK-RMSNorm attention (Sec. 3, Eqs. 1-10). The backbone is pre-trained with masked
patch reconstruction plus a Siamese similarity term between the overlapping patches
of a window and of the same series shifted forward (official ``core/loss.py``), then
frozen while a flatten-linear forecasting head is trained.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def patch_count(seq_len: int, patch_size: int, stride: int) -> int:
    """``K = floor((L - P) / S) + 1`` patches taken from the end of the window."""
    return (seq_len - patch_size) // stride + 1


def to_patches(x: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """``[B, L, C] -> [B, K, C, P]`` using the last ``P + S (K - 1)`` steps."""
    count = patch_count(x.shape[1], patch_size, stride)
    used = patch_size + stride * (count - 1)
    return x[:, x.shape[1] - used :].unfold(1, patch_size, stride)


def random_patch_mask(batch: int, patches: int, channels: int, ratio: float, device) -> torch.Tensor:
    """Binary mask ``[B, K, C]`` (1 = masked) hiding ``K - int(K (1 - ratio))`` random patches per series."""
    keep = int(patches * (1 - ratio))
    order = torch.rand(batch, patches, channels, device=device).argsort(dim=1)
    rank = order.argsort(dim=1)
    return (rank >= keep).float()


class Attention(nn.Module):
    """Eqs. (2)-(7): bias-free multi-head attention with RMSNorm on queries and keys."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, qk_norm: bool, bias: bool) -> None:
        super().__init__()
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-5) if qk_norm else nn.Identity()

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.n_heads, -1)).transpose(1, 2)  # [N, H, K, d_k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.q_norm(self._heads(self.q_proj(x)))
        key = self.k_norm(self._heads(self.k_proj(x)))
        value = self._heads(self.v_proj(x))
        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, scale=self.scale
        )
        return self.out_proj(out.transpose(1, 2).flatten(-2))


class EncoderLayer(nn.Module):
    """Pre-norm (Fig. 1) or post-norm layer: RMSNorm, attention, residual, RMSNorm, GELU FFN, residual."""

    def __init__(self, d_model, n_heads, d_ff, attn_dropout, ffn_dropout, pre_norm, qk_norm, bias) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = Attention(d_model, n_heads, attn_dropout, qk_norm, bias)
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.attention_dropout = nn.Dropout(ffn_dropout)
        # Eqs. (8)-(9): two bias-free linear maps with GELU.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ff, d_model, bias=bias),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attention_dropout(self.attention(self.norm1(x)))
            return x + self.ffn(self.norm2(x))  # Eq. (10)
        x = self.norm1(x + self.attention_dropout(self.attention(x)))
        return self.norm2(x + self.ffn(x))


class Backbone(nn.Module):
    """Eq. (1) patch projection plus positional table, ``e_layers`` encoder layers, final RMSNorm."""

    def __init__(self, patches, patch_size, d_model, n_heads, e_layers, d_ff, attn_dropout,
                 ffn_dropout, pre_norm, qk_norm, bias) -> None:
        super().__init__()
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.position = nn.Parameter(torch.empty(patches, d_model).uniform_(-0.02, 0.02))
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, attn_dropout, ffn_dropout, pre_norm, qk_norm, bias)
            for _ in range(e_layers)
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-5)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``[B, K, C, P] -> [B, C, d_model, K]`` (channel-independent)."""
        batch, count, channels, size = patches.shape
        x = patches.permute(0, 2, 1, 3).reshape(batch * channels, count, size)
        x = self.patch_proj(x) + self.position
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.reshape(batch, channels, count, -1).transpose(2, 3)


class Model(nn.Module):
    """SiamTST forecaster with its Siamese masked pre-training stage."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_size: int = 16,
        stride: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 4,
        d_ff: int = 256,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        pre_norm: bool = True,
        qk_norm: bool = True,
        bias: bool = False,
        min_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.55,
        alpha: float = 0.2,
        pretrain_epochs: int = 30,
        pretrain_lr: float = 1e-3,
        pretrain_weight_decay: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if seq_len < patch_size:
            raise ValueError("seq_len must be at least patch_size")
        if pretrain_epochs > 0 and (pred_len < stride or patch_count(seq_len, patch_size, stride) < 2):
            raise ValueError("pre-training needs pred_len >= stride and at least two patches")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_size = patch_size
        self.stride = stride
        self.patches = patch_count(seq_len, patch_size, stride)
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_weight_decay = pretrain_weight_decay
        self.freeze_backbone = freeze_backbone
        self.revin = RevIN(enc_in, affine=False)
        self.pair_revin = RevIN(enc_in, affine=False)
        self.backbone = Backbone(
            self.patches, patch_size, d_model, n_heads, e_layers, d_ff,
            attn_dropout, ffn_dropout, pre_norm, qk_norm, bias,
        )
        # Pre-training head: per-patch reconstruction; forecasting head: flatten + linear.
        self.pretrain_head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, patch_size))
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2), nn.Dropout(head_dropout), nn.Linear(d_model * self.patches, pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        latent = self.backbone(to_patches(normalized, self.patch_size, self.stride))
        forecast = self.head(latent).transpose(1, 2)  # [B, H, C]
        return self.revin(forecast, "denorm")

    def shift_patches(self, future_len: int) -> int:
        """Patches by which the Siamese view is shifted: half the window (official) or less if the future is short."""
        return min(self.patches // 2, future_len // self.stride)

    def pretraining_loss(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """``(1 - alpha) * masked MSE + alpha * (1 - cosine)`` (official ``PretrainLoss``).

        The second view is the window shifted forward by ``shift_patches`` patches
        (taken from ``x`` followed by ``future``), masked at the same patch indices
        and encoded without gradient; the similarity term compares the patches the
        two views share in time.
        """
        shift = self.shift_patches(future.shape[1])
        if shift < 1:
            raise ValueError("the Siamese view needs at least one stride of future values")
        steps = shift * self.stride
        view = torch.cat([x, future], dim=1)[:, steps : steps + self.seq_len]
        patches = to_patches(self.revin(x, "norm"), self.patch_size, self.stride)
        view_patches = to_patches(self.pair_revin(view, "norm"), self.patch_size, self.stride)
        batch, count, channels, _ = patches.shape
        ratio = self.min_mask_ratio + (self.max_mask_ratio - self.min_mask_ratio) * float(torch.rand(()))
        mask = random_patch_mask(batch, count, channels, ratio, x.device)
        keep = (1.0 - mask).unsqueeze(-1)
        latent = self.backbone(patches * keep)
        with torch.no_grad():
            view_latent = self.backbone(view_patches * keep)
        recon = self.pretrain_head(latent.transpose(2, 3)).permute(0, 2, 1, 3)  # [B, K, C, P]
        recon_loss = ((recon - patches) ** 2).mean(dim=-1)
        recon_loss = (recon_loss * mask).sum() / mask.sum().clamp_min(1.0)
        shared = count - shift
        similarity = F.cosine_similarity(
            latent[..., shift:].transpose(2, 3), view_latent[..., :shared].transpose(2, 3), dim=-1
        ).mean()
        return (1.0 - self.alpha) * recon_loss + self.alpha * (1.0 - similarity)

    def pretrain(self, train_loader, device) -> None:
        """Siamese masked pre-training on the training split, then freeze all but the forecasting head."""
        self.to(device)
        if self.pretrain_epochs > 0 and len(train_loader) > 0:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.pretrain_lr,
                weight_decay=self.pretrain_weight_decay, betas=(0.9, 0.98),
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.pretrain_lr, epochs=self.pretrain_epochs,
                steps_per_epoch=len(train_loader),
            )
            self.train()
            for _ in range(self.pretrain_epochs):
                for batch in train_loader:
                    x = batch[0].float().to(device)
                    future = batch[1].float().to(device)[:, -self.pred_len :, : x.shape[-1]]
                    optimizer.zero_grad()
                    self.pretraining_loss(x, future).backward()
                    optimizer.step()
                    scheduler.step()
        if self.freeze_backbone:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("head."))

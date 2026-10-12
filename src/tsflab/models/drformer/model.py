"""DRFormer: dynamic sparse patch tokenizer, multi-scale max-pooled token groups,
a Transformer with group-aware rotary position encoding, and deconvolution fusion.

Independent implementation from Section 3 (Eqs. 1-15, Algorithm 1) of Ding, Chen,
Lan and Zhang, "DRFormer: Multi-Scale Transformer Utilizing Diverse Receptive
Fields for Long Time-Series Forecasting" (arXiv 2408.02279, CIKM 2024), after
reading the pinned official code (``ruixindingECNU/DRFormer`` at ``30dcae62``, no
license file) to resolve omissions; nothing is copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer
from tsflab.models._inputs import check_window

# Floor of the cosine-annealed pruning rate (official ``CosineDecay`` eta_min).
DEATH_RATE_FLOOR = 1e-3


def patch_count(seq_len: int, patch_len: int, stride: int) -> int:
    """N = floor((I - P) / S) + 2 patches after padding the end by one stride (Sec. 3.2.2)."""
    return (seq_len - patch_len) // stride + 2


def pooling_geometry(num_patches: int, kernel: int) -> tuple[int, int, int]:
    """Padding, pooled length and deconvolution output padding for kernel ``K``.

    Pooling uses non-overlapping windows of ``K`` patches (Eq. 5) with symmetric
    padding ``ceil((K - N mod K) / 2)`` so that ``ceil(N / K)`` tokens remain; the
    transposed convolution of Eq. (14) maps them back to exactly ``N`` positions.
    """
    remainder = num_patches % kernel
    padding = 0 if remainder == 0 else math.ceil((kernel - remainder) / 2)
    length = (num_patches + 2 * padding - kernel) // kernel + 1
    output_padding = num_patches - (length * kernel - 2 * padding)
    return padding, length, output_padding


def cosine_death_rate(step: int, total: float, initial: float) -> float:
    """Annealed pruning rate of Eq. (3): floor + (alpha - floor) (1 + cos(pi t / T)) / 2."""
    if total <= 0:
        return initial
    return DEATH_RATE_FLOOR + (initial - DEATH_RATE_FLOOR) * 0.5 * (
        1.0 + math.cos(math.pi * step / total)
    )


def _uniform(shape, generator: torch.Generator | None, like: torch.Tensor) -> torch.Tensor:
    """U[0, 1) draws on CPU (optionally seeded) moved to ``like``'s device and dtype."""
    return torch.rand(shape, generator=generator).to(like)


class DynamicSparseLinear(nn.Module):
    """Patch-to-token linear map with a learnable sparse indicator (Eq. 1, Sec. 3.2.3-3.2.5).

    The ``d_model`` output rows are split into ``groups`` groups; group ``i``
    (1-based) may only activate the last ``i * P / G`` input positions of a patch
    (its exploration region), so token dimensions see receptive fields of
    different sizes. The indicator ``I(w)`` is a buffer updated by
    :meth:`prune_and_regrow` (Algorithm 1), not by back-propagation.
    """

    def __init__(self, patch_len: int, d_model: int, groups: int, active_ratio: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.groups = groups
        self.rows_per_group = d_model // groups
        linear = nn.Linear(patch_len, d_model)
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone())
        step = patch_len // groups
        window = torch.arange(1, groups + 1).repeat_interleave(self.rows_per_group) * step
        self.register_buffer("window", window, persistent=False)
        position = torch.arange(patch_len)
        active = torch.ceil(window.double() * active_ratio).long()
        mask = (position[None, :] >= patch_len - active[:, None]).float()
        self.register_buffer("mask", mask)
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def candidate_region(self) -> torch.Tensor:
        """Boolean ``[d_model, P]`` map of the per-group exploration regions C."""
        position = torch.arange(self.patch_len, device=self.window.device)
        return position[None, :] >= self.patch_len - self.window[:, None]

    def masked_weight(self) -> torch.Tensor:
        """``w_E ⊙ I(w_E)`` of Eq. (1)."""
        return self.weight * self.mask

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return F.linear(patches, self.masked_weight(), self.bias)

    @torch.no_grad()
    def prune_and_regrow(
        self, death_rate: float, generator: torch.Generator | None = None
    ) -> tuple[int, int]:
        """One indicator update of Algorithm 1 (lines 6-9), group by group.

        Per group, the ``n = ceil(death_rate * active)`` active weights with the
        smallest magnitude are deactivated, then ``n`` inactive positions of the
        group's exploration region are activated at random (the just-pruned ones
        are eligible). Regrown positions get a fresh ``kaiming_uniform`` draw
        (bound ``1 / sqrt(P)``) added to their masked value. Returns the
        ``(pruned, grown)`` totals.
        """
        weight = self.weight * self.mask  # inactive weights are zero, as after every step
        region = self.candidate_region()
        old_mask = self.mask.bool()
        new_mask = old_mask.clone()
        kept = old_mask.clone()
        pruned = grown = 0
        for group in range(self.groups):
            rows = slice(group * self.rows_per_group, (group + 1) * self.rows_per_group)
            active = old_mask[rows].flatten()
            n = math.ceil(death_rate * int(active.sum()))
            if n == 0:
                continue
            magnitude = weight[rows].abs().flatten().masked_fill(~active, float("inf"))
            survivors = active.clone()
            survivors[magnitude.topk(n, largest=False).indices] = False
            # Random growth inside the region first; outside it only if the region
            # has fewer than n free slots (unreachable from the default indicator).
            score = _uniform(survivors.shape, generator, weight) + 10.0 * region[rows].flatten()
            score = score.masked_fill(survivors, -float("inf"))
            final = survivors.clone()
            final[score.topk(n).indices] = True
            kept[rows] = survivors.view(self.rows_per_group, self.patch_len)
            new_mask[rows] = final.view(self.rows_per_group, self.patch_len)
            pruned += n
            grown += n
        bound = 1.0 / math.sqrt(self.patch_len)
        fresh = (2.0 * _uniform(weight.shape, generator, weight) - 1.0) * bound
        # Every regrown position (including one pruned in this same update, which
        # keeps its masked value as in the official order) receives a fresh draw.
        self.weight.copy_(weight * new_mask + fresh * (new_mask & ~kept))
        self.mask.copy_(new_mask.float())
        return pruned, grown


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal 2-D rotations (Eqs. 7, 9) to adjacent feature pairs of ``[B, S, H, E]``."""
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class GroupAwareRoPEAttention(nn.Module):
    """Multi-head attention with group-aware rotary position encoding (Sec. 3.4.1, Eqs. 7-11).

    Queries and keys are rotated twice: by the intra-group angle ``m_hat * theta``
    (position aligned to the base patch grid) and by the inter-group angle
    ``i * theta`` (group index). Scores are ``Q_inter K_inter^T + Q_intra K_intra^T``
    scaled by ``1 / sqrt(d_k)`` (Eq. 11), computed with the shared full-attention
    core over the concatenated rotated features.
    """

    def __init__(self, d_model: int, n_heads: int, group_lengths: list[int], dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.inner_attention = FullAttention(
            mask_flag=False, scale=1.0 / math.sqrt(head_dim), attention_dropout=dropout
        )
        theta = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)  # Eq. (8)
        base = group_lengths[0]
        intra = torch.cat([torch.arange(n, dtype=torch.float64) * base / n for n in group_lengths])
        inter = torch.cat(
            [torch.full((n,), float(i), dtype=torch.float64) for i, n in enumerate(group_lengths)]
        )
        for name, position in (("intra", intra), ("inter", inter)):
            angle = position[:, None] * theta[None, :]
            self.register_buffer(f"{name}_cos", angle.cos().float(), persistent=False)
            self.register_buffer(f"{name}_sin", angle.sin().float(), persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """``[R_inter x, R_intra x]`` along the head feature axis."""
        return torch.cat(
            (rotary(x, self.inter_cos, self.inter_sin), rotary(x, self.intra_cos, self.intra_sin)),
            dim=-1,
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        del attn_mask, tau, delta
        batch, length, _ = queries.shape
        q = self.query_projection(queries).view(batch, length, self.n_heads, -1)
        k = self.key_projection(keys).view(batch, keys.shape[1], self.n_heads, -1)
        v = self.value_projection(values).view(batch, values.shape[1], self.n_heads, -1)
        out, attn = self.inner_attention(self.rotate(q), self.rotate(k), v, None)
        return self.out_projection(out.reshape(batch, length, -1)), attn


class Model(nn.Module):
    """DRFormer forecaster; channel-independent over ``[batch, seq_len, enc_in]``."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        e_layers: int = 2,
        patch_len: int = 16,
        stride: int = 4,
        sequence_num: int = 3,
        dropout: float = 0.05,
        activation: str = "gelu",
        mask_groups: int = 8,
        active_ratio: float = 0.5,
        death_rate: float = 0.5,
        update_frequency: float = 0.3,
        mask_epochs: float = 5.0,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        self.death_rate = death_rate
        self.update_frequency = update_frequency
        self.mask_epochs = mask_epochs
        self.num_patches = patch_count(seq_len, patch_len, stride)
        self.kernels = [2**j for j in range(1, sequence_num)]
        geometry = [pooling_geometry(self.num_patches, k) for k in self.kernels]
        self.paddings = [g[0] for g in geometry]
        self.group_lengths = [self.num_patches] + [g[1] for g in geometry]

        self.revin = RevIN(enc_in, eps=1e-5, affine=False)
        self.tokenizer = DynamicSparseLinear(patch_len, d_model, mask_groups, active_ratio)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    GroupAwareRoPEAttention(d_model, n_heads, self.group_lengths, dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.deconvs = nn.ModuleList(
            nn.ConvTranspose1d(
                d_model, d_model, kernel_size=k, stride=k, padding=p,
                output_padding=op, bias=False,
            )
            for k, (p, _, op) in zip(self.kernels, geometry)
        )
        self.head = FlattenForecastHead(
            individual=False, n_vars=enc_in, nf=d_model * self.num_patches,
            target_window=pred_len, head_dropout=dropout,
        )
        # Dynamic-sparse-training schedule state (spec.training_setup / training_objective).
        self.register_buffer("steps_per_epoch", torch.zeros((), dtype=torch.long))
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

    # -------------------------------------------------------------- tokenizer
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, L] -> [B, C, N, P]``: replicate-pad the end by one stride, then unfold (Sec. 3.2.2)."""
        x = F.pad(x, (0, self.stride), mode="replicate")
        return x.unfold(-1, self.patch_len, self.stride)

    def multi_scale(self, tokens: torch.Tensor) -> torch.Tensor:
        """Concatenate the base tokens with hierarchical max-pooled groups ``K = 2, 4, ...`` (Eqs. 4-5)."""
        seq = tokens.transpose(1, 2)
        pooled = [
            F.max_pool1d(seq, k, stride=k, padding=p).transpose(1, 2)
            for k, p in zip(self.kernels, self.paddings)
        ]
        return torch.cat([tokens, *pooled], dim=1)

    def fuse(self, encoded: torch.Tensor) -> torch.Tensor:
        """Split per group, upsample each pooled group by deconvolution and sum (Eqs. 13-15)."""
        groups = torch.split(encoded, self.group_lengths, dim=1)
        fused = groups[0]
        for group, deconv in zip(groups[1:], self.deconvs):
            fused = fused + deconv(group.transpose(1, 2)).transpose(1, 2)
        return fused

    # ------------------------------------------------------- sparse schedule
    def death_rate_at(self, step: int) -> float:
        total = int(self.steps_per_epoch) * self.mask_epochs
        return cosine_death_rate(step, total, self.death_rate)

    def advance_sparse_schedule(self) -> bool:
        """Run Algorithm 1's indicator update when the completed step count calls for one.

        Every ``floor(update_frequency * steps_per_epoch)`` optimizer steps during
        the first ``mask_epochs`` epochs; afterwards (and when no training setup
        recorded the epoch length) the indicator stays fixed.
        """
        per_epoch = int(self.steps_per_epoch)
        step = int(self.train_steps)
        if per_epoch == 0 or step == 0:
            return False
        every = max(1, int(self.update_frequency * per_epoch))
        if step % every or (step - 1) // per_epoch >= self.mask_epochs:
            return False
        self.tokenizer.prune_and_regrow(self.death_rate_at(step))
        return True

    # ---------------------------------------------------------------- forward
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        x = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        tokens = self.tokenizer(self.patchify(x))  # [B, C, N, D]  (Eq. 1)
        tokens = tokens.reshape(batch * self.enc_in, self.num_patches, -1)
        encoded, _ = self.encoder(self.multi_scale(tokens))
        fused = self.fuse(encoded).reshape(batch, self.enc_in, self.num_patches, -1)
        out = self.head(fused.transpose(2, 3)).transpose(1, 2)  # [B, H, C]
        return self.revin(out, "denorm")

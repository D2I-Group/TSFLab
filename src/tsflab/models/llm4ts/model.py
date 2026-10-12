"""Local LLM4TS implementation from the paper and pinned official code.

LLM4TS adapts the first layers of a pretrained GPT-2 to patched, channel-
independent time series in two stages: (1) time-series alignment, an
autoregressive next-patch objective with LayerNorm tuning and LoRA, and
(2) forecasting fine-tuning with RevIN and a flatten-linear head, trained by
linear probing followed by fine-tuning (LP-FT).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import TemporalEmbedding, TokenEmbedding
from tsflab.models._components.gpt2_backbone import GPT2Backbone, GPT2Config, load_gpt2_weights
from tsflab.models._components.marks import adapt_tslib_marks
from tsflab.models._components.revin import RevIN


class LoRALinear(nn.Module):
    """``W x + (alpha / r) * B A dropout(x)`` around a frozen base projection.

    ``A`` uses Kaiming-uniform initialization (``a = sqrt(5)``) and ``B`` starts
    at zero, so the wrapped layer initially equals the base layer.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class PatchEncoding(nn.Module):
    """Token encoding of patches plus the "select first" multi-scale temporal encoding.

    Paper Eq. 3 and 5. Positions (Eq. 4) are GPT-2's own trainable ``wpe`` table.
    """

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float,
                 token_embed_type: str, temporal_embed_type: str, freq: str) -> None:
        super().__init__()
        self.patch_len, self.stride, self.freq = patch_len, stride, freq
        if token_embed_type == "conv":
            # 1-D convolution (kernel 3, circular) across neighbouring patches.
            self.token = TokenEmbedding(patch_len, d_model)
        else:  # linear
            self.token = nn.Linear(patch_len, d_model, bias=False)
        self.temporal = (
            None if temporal_embed_type == "none"
            else TemporalEmbedding(d_model, embed_type=temporal_embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(dropout)

    def patches(self, series: torch.Tensor) -> torch.Tensor:
        """``[N, L] -> [N, T_p, P]``: replicate-pad ``stride`` steps at the end, then unfold."""
        padded = torch.cat([series, series[:, -1:].expand(-1, self.stride)], dim=1)
        return padded.unfold(-1, self.patch_len, self.stride)

    def calendar(self, x_mark: torch.Tensor, channels: int, num_patches: int) -> torch.Tensor:
        """Level 1 (sum of per-attribute tables) and Level 2 ("select first") aggregation."""
        marks = adapt_tslib_marks(x_mark, embed_type="learned", freq=self.freq)
        if self.freq == "t":
            # 15-minute bins, as in the categorical Time-Series-Library marks.
            marks = torch.cat([marks[..., :4], torch.div(marks[..., 4:5], 15, rounding_mode="floor")], -1)
        padded = torch.cat([marks, marks[:, -1:].expand(-1, self.stride, -1)], dim=1)
        first = padded[:, torch.arange(num_patches, device=marks.device) * self.stride]
        embedded = self.temporal(first)  # [B, T_p, D]
        return embedded.repeat_interleave(channels, dim=0)

    def forward(self, values: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        """``values [B, L, C]`` (normalized) -> tokens ``[B * C, T_p, D]``."""
        batch, length, channels = values.shape
        series = values.permute(0, 2, 1).reshape(batch * channels, length)
        tokens = self.token(self.patches(series))
        if self.temporal is not None and x_mark is not None:
            tokens = tokens + self.calendar(x_mark, channels, tokens.shape[1])
        return self.dropout(tokens)


def _check_config(seq_len, pred_len, patch_len, stride) -> None:
    if patch_len > seq_len + stride:
        raise ValueError("patch_len must not exceed seq_len + stride")


def _batch_to(batch, device):
    """``(x, y, x_mark)`` of a loader batch as float tensors on ``device`` (``None`` kept)."""
    return (t.float().to(device) if t is not None else None for t in batch[:3])


class Model(nn.Module):
    """LLM4TS: GPT-2 (first ``first_k_layers`` blocks) aligned and fine-tuned for forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        first_k_layers: int = 6,
        dropout: float = 0.05,
        token_embed_type: str = "conv",
        temporal_embed_type: str = "learned",
        freq: str = "h",
        lora_r: int = 8,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        align_epochs: int = 5,
        align_lr: float = 7.912045141879411e-05,
        align_weight_decay: float = 0.0005542494992024964,
        probe_epochs: int = 7,
        probe_lr: float = 1.8257759510439175e-05,
        probe_weight_decay: float = 0.0014555863788252605,
    ) -> None:
        super().__init__()
        _check_config(seq_len, pred_len, patch_len, stride)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        self.num_patches = (seq_len + stride - patch_len) // stride + 1
        self.align_epochs, self.align_lr, self.align_weight_decay = align_epochs, align_lr, align_weight_decay
        self.probe_epochs, self.probe_lr, self.probe_weight_decay = probe_epochs, probe_lr, probe_weight_decay

        self.llm = GPT2Backbone(GPT2Config(n_layer=first_k_layers))
        width = self.llm.config.n_embd
        if self.num_patches > self.llm.config.n_positions:
            raise ValueError("too many patches for the GPT-2 context")
        for block in self.llm.h:
            block.attn.c_attn = LoRALinear(block.attn.c_attn, lora_r, lora_alpha, lora_dropout)
        self.revin = RevIN(1, affine=True)
        self.encoding = PatchEncoding(patch_len, stride, width, dropout,
                                      token_embed_type, temporal_embed_type, freq)
        # Stage 1 output layer W_tsa (Eq. 8) and stage 2 flatten head W_fft (Eq. 11).
        self.align_head = nn.Sequential(nn.Linear(width, patch_len, bias=False), nn.Dropout(dropout))
        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * width, pred_len, bias=False),
            nn.Dropout(dropout),
        )
        # Frozen sinusoidal calendar tables never become trainable.
        self._fixed = {id(p) for p in self.parameters() if not p.requires_grad}
        self.set_stage("finetune")

    # ----------------------------------------------------------------- stages
    def set_stage(self, stage: str) -> None:
        """Select the trainable parameter set of one training stage.

        ``align``: encodings, alignment head, GPT-2 LayerNorms and positions, LoRA.
        ``probe``: the forecasting head only (linear probing).
        ``finetune``: RevIN affine, encodings, LoRA, and the forecasting head.
        """
        groups = {
            "align": ("encoding.", "align_head.", "lora_", ".ln_", "llm.ln_f.", "llm.wpe."),
            "probe": ("forecast_head.",),
            "finetune": ("revin.", "encoding.", "lora_", "forecast_head."),
        }
        if stage not in groups:
            raise ValueError(f"unknown LLM4TS stage {stage!r}")
        for name, parameter in self.named_parameters():
            trainable = id(parameter) not in self._fixed and any(
                name.startswith(key) or key in name for key in groups[stage]
            )
            parameter.requires_grad_(trainable)

    def load_llm_weights(self, path) -> None:
        """Load released GPT-2 weights (first ``first_k_layers`` blocks) under the LoRA wrappers."""
        released = load_gpt2_weights(GPT2Backbone(self.llm.config), path).state_dict()
        state = {name.replace("attn.c_attn.", "attn.c_attn.base."): tensor
                 for name, tensor in released.items()}
        missing, unexpected = self.llm.load_state_dict(state, strict=False)
        if unexpected or any("lora_" not in name for name in missing):
            raise KeyError(f"GPT-2 weights do not fit LLM4TS: missing={missing}, unexpected={unexpected}")

    # ---------------------------------------------------------------- forward
    def _hidden(self, normalized: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        return self.llm(self.encoding(normalized, x_mark))  # Eq. 6-7

    def align_forward(self, normalized: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        """Stage 1: ``[B, L, C]`` -> next-patch predictions ``[B * C, T_p, P]`` (Eq. 8)."""
        return self.align_head(self._hidden(normalized, x_mark))

    def alignment_loss(self, x_enc: torch.Tensor, future: torch.Tensor,
                       x_mark: torch.Tensor | None = None) -> torch.Tensor:
        """Stage 1 objective (Eq. 9): MSE against the patches shifted by one stride.

        The window and its next ``stride`` values are standardized together
        without affine parameters (Eq. 1); the target is that series shifted by
        ``stride`` and patched like the input.
        """
        if future.shape[1] < self.stride:
            raise ValueError("the alignment stage needs at least `stride` future values")
        series = torch.cat([x_enc, future[:, : self.stride]], dim=1)
        mean = series.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(series.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        series = (series - mean) / std
        inputs, shifted = series[:, : self.seq_len], series[:, self.stride:]
        batch, length, channels = shifted.shape
        target = self.encoding.patches(shifted.permute(0, 2, 1).reshape(batch * channels, length))
        return F.mse_loss(self.align_forward(inputs, x_mark), target)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        """Stage 2 forecast (Eq. 10-11): RevIN, CI patches, GPT-2, flatten head, RevIN denorm."""
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"LLM4TS expects [batch, {self.seq_len}, channels]")
        batch, length, channels = x_enc.shape
        series = x_enc.permute(0, 2, 1).reshape(batch * channels, length, 1)
        normalized = self.revin(series, "norm").reshape(batch, channels, length).permute(0, 2, 1)
        forecast = self.forecast_head(self._hidden(normalized, x_mark_enc))  # [B * C, pred_len]
        forecast = self.revin(forecast.unsqueeze(-1), "denorm")
        return forecast.reshape(batch, channels, self.pred_len).permute(0, 2, 1)

    # --------------------------------------------------------- staged training
    def pretrain(self, train_loader, device) -> None:
        """Run time-series alignment, then linear probing; leave the fine-tuning set trainable.

        The trainer's ordinary forecasting loop afterwards is the fine-tuning
        half of LP-FT. Both stages here use MSE and AdamW, as in the official code.
        """
        self.to(device)

        def run(stage: str, epochs: int, lr: float, weight_decay: float, step) -> None:
            if epochs == 0:
                return
            self.set_stage(stage)
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            self.train()
            for _ in range(epochs):
                for batch in train_loader:
                    x, y, x_mark = _batch_to(batch, device)
                    optimizer.zero_grad()
                    step(x, y, x_mark).backward()
                    optimizer.step()

        run("align", self.align_epochs, self.align_lr, self.align_weight_decay,
            lambda x, y, m: self.alignment_loss(x, y[:, -self.pred_len:], m))
        run("probe", self.probe_epochs, self.probe_lr, self.probe_weight_decay,
            lambda x, y, m: F.mse_loss(self(x, m), y[:, -self.pred_len:]))
        self.set_stage("finetune")

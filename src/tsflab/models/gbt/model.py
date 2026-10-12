"""GBT: two-stage Transformer framework with a Good Beginning.

Independent implementation from Section 4 (Figs. 4-5, Eq. 3, Algorithm 1) of
Shen, Wei and Wang, "GBT: Two-stage transformer framework for non-stationary time
series forecasting" (arXiv 2307.08302, Neural Networks 2023), after reading the
pinned official code (``OrigamiSL/GBT`` at ``4146c386``, Apache-2.0) to resolve
omissions; nothing is copied.

Stage 1 (Auto-Regression) maps the history to a first forecast with pyramid
attention + convolution blocks and a flatten-linear head. It is trained alone by
:meth:`Model.pretrain` and then frozen. Stage 2 (Self-Regression) re-reads that
detached forecast with masked self-attention decoders whose scores carry the
Error Score Modification (ESM) Gaussian prior, and adds a correction to it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from tsflab.models._components.embed import PositionalEmbedding
from tsflab.models._components.masking import TriangularCausalMask


def halved_length(length: int) -> int:
    """Sequence length after one stride-2 residual unit (pool 3/2/1 or odd-kernel
    conv with ``(k - 1) // 2`` padding): ``ceil(length / 2)``."""
    return (length + 1) // 2


def esm_prior(sigma_logits: torch.Tensor, length: int) -> torch.Tensor:
    """Error Score Modification prior (Eq. 3, Algorithm 1).

    ``sigma_logits`` is ``[B, H, L]`` (one scale per head and query position). The
    scale is squashed as ``s = sigmoid(5 x) + 1e-5`` and mapped to ``3 ** s - 1``
    (the official form; Algorithm 1 prints ``3 s - 1``). The returned ``[B, H, L, L]``
    term is the zero-centred Gaussian density of the key position ``j``,
    ``N(j; 0, sigma_i^2)``, so earlier forecast steps receive more score.
    """
    sigma = torch.sigmoid(5.0 * sigma_logits) + 1e-5
    sigma = torch.pow(3.0, sigma) - 1.0
    sigma = sigma.unsqueeze(-1)
    position = torch.arange(length, device=sigma_logits.device, dtype=sigma.dtype)
    return torch.exp(-(position**2) / (2.0 * sigma**2)) / (math.sqrt(2.0 * math.pi) * sigma)


class PointwiseEmbedding(nn.Module):
    """Value embedding as a kernel-1 convolution with bias and Kaiming-normal
    (fan-in, leaky-ReLU) weights, optionally plus sinusoidal positions, then dropout."""

    def __init__(self, c_in: int, d_model: int, dropout: float, position: bool) -> None:
        super().__init__()
        self.value = nn.Conv1d(c_in, d_model, kernel_size=1)
        nn.init.kaiming_normal_(self.value.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.position = PositionalEmbedding(d_model) if position else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.position is not None:
            out = out + self.position(out)
        return self.dropout(out)


class ARSelfAttention(nn.Module):
    """Encoder self-attention of an Auto-Regression block: weight-normalized
    point-wise Q/K/V/output projections, unmasked scaled softmax, residual add.
    The encoder keeps no feed-forward layer (Section 4.1.2)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.query = weight_norm(nn.Conv1d(d_model, d_model, 1))
        self.key = weight_norm(nn.Conv1d(d_model, d_model, 1))
        self.value = weight_norm(nn.Conv1d(d_model, d_model, 1))
        self.out = weight_norm(nn.Conv1d(d_model, d_model, 1))
        self.dropout = nn.Dropout(dropout)

    def _heads(self, proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return proj(x.transpose(1, 2)).transpose(1, 2).reshape(b, length, self.n_heads, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        q, k, v = self._heads(self.query, x), self._heads(self.key, x), self._heads(self.value, x)
        scores = torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(q.size(-1))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, v).reshape(b, length, -1)
        return x + self.out(out.transpose(1, 2)).transpose(1, 2)


class ResidualConvUnit(nn.Module):
    """One residual unit of the ConvBlock (Fig. 5a).

    Main path: WN conv (``kernel``, ``stride``) -> dropout -> GELU -> WN conv
    (kernel 3, ``c_in -> c_out``) -> dropout -> GELU. Shortcut: a point-wise conv when
    the width changes, then a 3/2/1 max-pool when ``stride == 2`` (Res-P); with
    ``stride == 1`` and a width change it is Res-C.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int, dropout: float, stride: int) -> None:
        super().__init__()
        self.first = weight_norm(nn.Conv1d(c_in, c_in, kernel, stride=stride, padding=(kernel - 1) // 2))
        self.second = weight_norm(nn.Conv1d(c_in, c_out, 3, padding=1))
        self.dropout = nn.Dropout(dropout)
        self.shortcut_conv = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.shortcut_pool = nn.MaxPool1d(3, stride=2, padding=1) if stride != 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        shortcut = h
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        if self.shortcut_pool is not None:
            shortcut = self.shortcut_pool(shortcut)
        h = F.gelu(self.dropout(self.first(h)))
        h = F.gelu(self.dropout(self.second(h)))
        return (h + shortcut).transpose(1, 2)


class ARBlock(nn.Module):
    """Auto-Regression block: encoder self-attention followed by a ConvBlock that
    halves the length and doubles the width (Fig. 5a: Res-P then Res-C, or one fused
    stride-2 widening unit as in the official default extractor)."""

    def __init__(
        self, c_in: int, n_heads: int, kernel: int, dropout: float, fused_conv: bool
    ) -> None:
        super().__init__()
        self.attention = ARSelfAttention(c_in, n_heads, dropout)
        if fused_conv:
            units = [ResidualConvUnit(c_in, 2 * c_in, kernel, dropout, stride=2)]
        else:
            units = [
                ResidualConvUnit(c_in, c_in, kernel, dropout, stride=2),
                ResidualConvUnit(c_in, 2 * c_in, kernel, dropout, stride=1),
            ]
        self.conv = nn.Sequential(*units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.attention(x))


class ARPyramidBranch(nn.Module):
    """One pyramid network (Fig. 5b): embedding, ``blocks`` AR blocks, and an FC
    layer from the flattened feature map to ``pred_len x c_out`` (replacing
    cross-attention)."""

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        c_in: int,
        c_out: int,
        fd_model: int,
        blocks: int,
        n_heads: int,
        kernel: int,
        dropout: float,
        fused_conv: bool,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.c_out = c_out
        self.embedding = PointwiseEmbedding(c_in, fd_model, dropout, position=False)
        self.blocks = nn.ModuleList(
            ARBlock(fd_model * 2**i, n_heads, kernel, dropout, fused_conv) for i in range(blocks)
        )
        length = input_len
        for _ in range(blocks):
            length = halved_length(length)
        self.head = nn.Linear(fd_model * 2**blocks * length, pred_len * c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x[:, -self.input_len :])
        for block in self.blocks:
            h = block(h)
        out = self.head(h.transpose(1, 2).flatten(1))
        return out.view(-1, self.c_out, self.pred_len).transpose(1, 2)


class AutoRegressionStage(nn.Module):
    """First stage: average of ``pyramid`` branches; branch ``i`` reads the last
    ``ar_len // 2**i`` steps through ``blocks - i`` AR blocks."""

    def __init__(
        self,
        ar_len: int,
        pred_len: int,
        c_in: int,
        c_out: int,
        fd_model: int,
        blocks: int,
        pyramid: int,
        n_heads: int,
        kernel: int,
        dropout: float,
        fused_conv: bool,
    ) -> None:
        super().__init__()
        if ar_len // 2 ** (pyramid - 1) < 1:
            raise ValueError("ar_len is too short for the pyramid depth")
        self.branches = nn.ModuleList(
            ARPyramidBranch(
                ar_len // 2**i, pred_len, c_in, c_out, fd_model, blocks - i,
                n_heads, kernel, dropout, fused_conv,
            )
            for i in range(pyramid)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([branch(x) for branch in self.branches]).mean(0)


class ESMSelfAttention(nn.Module):
    """Masked self-attention with ESM (Algorithm 1): ``softmax(scale * mask(QK^T + G))V``,
    ``G`` from :func:`esm_prior` with one learnable scale per head and query.

    With ``mix`` the ``[B, L, H, D]`` context is read in head-major order before the
    output projection (official default of the Informer-style layer)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, mix: bool) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.mix = mix
        self.sigma = nn.Linear(d_model, n_heads)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Masked, ESM-modified, unscaled scores ``[B, H, L, L]``."""
        b, length, _ = x.shape
        q = self.query(x).view(b, length, self.n_heads, -1)
        k = self.key(x).view(b, length, self.n_heads, -1)
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        scores = scores + esm_prior(self.sigma(x).transpose(1, 2), length)
        mask = TriangularCausalMask(b, length, device=x.device).mask
        return scores.masked_fill(mask, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, d_model = x.shape
        head_dim = d_model // self.n_heads
        v = self.value(x).view(b, length, self.n_heads, head_dim)
        attn = self.dropout(torch.softmax(self.scores(x) / math.sqrt(head_dim), dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, v)
        if self.mix:
            out = out.transpose(1, 2).contiguous()
        return self.out(out.reshape(b, length, d_model))


class SRDecoderLayer(nn.Module):
    """Self-Regression decoder layer: ESM masked self-attention (no cross-attention),
    post-norm, point-wise feed-forward (``4 * d_model``), post-norm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, mix: bool) -> None:
        super().__init__()
        self.attention = ESMSelfAttention(d_model, n_heads, dropout, mix)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, 4 * d_model)
        self.ffn_out = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_attn(x + self.dropout(self.attention(x)))
        y = self.dropout(F.gelu(self.ffn_in(x)))
        y = self.dropout(self.ffn_out(y))
        return self.norm_ffn(x + y)


class Model(nn.Module):
    """GBT forecaster ``[B, seq_len, enc_in] -> [B, pred_len, enc_in]``.

    ``forward`` is the second-stage output: first-stage forecast (detached) plus the
    Self-Regression correction. Calendar marks and decoder inputs are ignored.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        ar_len: int | None = None,
        fd_model: int = 32,
        d_model: int = 512,
        n_heads: int = 8,
        ar_blocks: int = 3,
        pyramid: int = 3,
        d_layers: int = 2,
        kernel: int = 3,
        dropout: float = 0.1,
        channel_independent: bool = True,
        fused_conv: bool = False,
        mix: bool = True,
        ar_epochs: int = 1,
        ar_lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.ar_len = seq_len if ar_len is None else ar_len
        if not 1 <= self.ar_len <= seq_len:
            raise ValueError("ar_len must be in [1, seq_len]")
        self.channel_independent = channel_independent
        self.ar_epochs = ar_epochs
        self.ar_lr = ar_lr
        width = 1 if channel_independent else enc_in
        self.auto_regression = AutoRegressionStage(
            self.ar_len, pred_len, width, width, fd_model, ar_blocks, pyramid,
            n_heads, kernel, dropout, fused_conv,
        )
        self.embedding = PointwiseEmbedding(width, d_model, dropout, position=True)
        self.decoder = nn.ModuleList(
            SRDecoderLayer(d_model, n_heads, dropout, mix) for _ in range(d_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, width)
        self._auto_regression_frozen = False

    def _to_units(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L, C] -> [B * C, L, 1]`` in channel-independent mode."""
        if not self.channel_independent:
            return x
        b, length, c = x.shape
        return x.permute(0, 2, 1).reshape(b * c, length, 1)

    def _from_units(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        if not self.channel_independent:
            return x
        return x.reshape(batch, self.enc_in, -1).permute(0, 2, 1)

    def first_stage(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Auto-Regression stage forecast ``[B, pred_len, enc_in]`` (the Good Beginning)."""
        self._check(x_enc)
        return self._from_units(self.auto_regression(self._to_units(x_enc)), x_enc.size(0))

    def second_stage(self, beginning: torch.Tensor) -> torch.Tensor:
        """Self-Regression stage on a detached first-stage forecast ``[B, pred_len, enc_in]``."""
        units = self._to_units(beginning.detach())
        h = self.embedding(units)
        for layer in self.decoder:
            h = layer(h)
        out = self.projection(self.norm(h)) + units
        return self._from_units(out, beginning.size(0))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.second_stage(self.first_stage(x_enc))

    def _check(self, x_enc: torch.Tensor) -> None:
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len or x_enc.size(2) != self.enc_in:
            raise ValueError(f"GBT expects [B, {self.seq_len}, {self.enc_in}]")

    def pretrain(self, train_loader, device) -> None:
        """Stage 1: train the Auto-Regression stage alone with MSE against the future
        window (Adam, learning rate halved after every epoch), then freeze it."""
        if self._auto_regression_frozen:
            return
        self.to(device)
        self.auto_regression.train()
        optimizer = torch.optim.Adam(self.auto_regression.parameters(), lr=self.ar_lr)
        for epoch in range(self.ar_epochs):
            for group in optimizer.param_groups:
                group["lr"] = self.ar_lr * 0.5**epoch
            for batch in train_loader:
                x = batch[0].float().to(device)
                y = batch[1].float().to(device)[:, -self.pred_len :, : self.enc_in]
                optimizer.zero_grad()
                F.mse_loss(self.first_stage(x), y).backward()
                optimizer.step()
        for parameter in self.auto_regression.parameters():
            parameter.requires_grad_(False)
        self._auto_regression_frozen = True

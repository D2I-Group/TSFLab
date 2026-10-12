"""Local ReNF implementation (paper Sections 2.2-2.4, Definition 2.3, Eqs. 3-4, Fig. 3).

Independent rewrite from the paper; the pinned official repository (MIT) was
read only to resolve omissions such as the block layouts of the two variants,
the dense concatenation of all previous sub-forecasts, the dropout schedules,
the learnable additive position table, and the block-wise loss weights.
"""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN

#: Dropout applied to the raw window before the first block's projection.
FIRST_BLOCK_INPUT_DROPOUT = 0.1
#: Scale gamma of the block-wise loss weights gamma / n (Eq. 4).
LOSS_SCALE = 20.0


def concat_width(seq_len: int, segment: int, block_index: int) -> int:
    """Input width of block ``k`` (0-based): window plus all previous sub-forecasts.

    Previous sub-forecasts have lengths ``segment * 1, ..., segment * k``.
    """
    return seq_len + segment * block_index * (block_index + 1) // 2


def time_frequency_l1(forecast: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Hybrid loss of Eq. (4) for one sub-forecast ``[B, h, C]``.

    ``alpha * mean |rfft(y_hat) - rfft(y)| + (1 - alpha) * mean |y_hat - y|``
    with the FFT along time.
    """
    time_term = (forecast - target).abs().mean()
    if alpha == 0.0:
        return time_term
    freq_term = (torch.fft.rfft(forecast, dim=1) - torch.fft.rfft(target, dim=1)).abs().mean()
    return alpha * freq_term + (1.0 - alpha) * time_term


def block_supervision_loss(
    forecasts: list[torch.Tensor], target: torch.Tensor, alpha: float, scale: float = LOSS_SCALE
) -> torch.Tensor:
    """Eq. (4): sum over sub-forecasts of ``(scale / n) * hybrid(Y_hat_n, Y[:h_n])``."""
    loss = forecasts[0].new_zeros(())
    for n, forecast in enumerate(forecasts, start=1):
        loss = loss + scale / n * time_frequency_l1(forecast, target[:, : forecast.shape[1]], alpha)
    return loss


class VariateNorm(nn.Module):
    """Non-affine normalization of every time position across variates.

    Input ``[B, V, T]``; each ``[b, :, t]`` slice is standardized with its
    biased variance (official ``norm_name='instance'`` pre-normalization).
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 2:
            raise ValueError("variate normalization needs at least two variates")
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class FirstBlock(nn.Module):
    """Sub-forecaster 1: Drop(0.1) -> Linear(T, d_ff) -> Linear(no bias) -> GELU -> Drop -> head."""

    def __init__(self, seq_len: int, segment: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(FIRST_BLOCK_INPUT_DROPOUT)
        self.projection = nn.Linear(seq_len, d_ff)
        self.transform = nn.Linear(d_ff, d_ff, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_ff, segment)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rep = self.dropout(self.activation(self.transform(self.projection(self.input_dropout(x)))))
        return self.head(rep), rep


class AlphaBlock(nn.Module):
    """ReNF-alpha sub-forecaster k > 1: LayerNorm -> Drop -> Linear -> Linear -> Drop -> act -> head.

    The activation is GELU except in the last block, which stays linear.
    """

    def __init__(
        self, width: int, horizon: int, d_ff: int, dropout: float, input_dropout: float, linear: bool
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.input_dropout = nn.Dropout(input_dropout)
        self.projection = nn.Linear(width, d_ff)
        self.transform = nn.Linear(d_ff, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Identity() if linear else nn.GELU()
        self.head = nn.Linear(d_ff, horizon)

    def forward(self, x: torch.Tensor, rep_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rep = self.projection(self.input_dropout(self.norm(x)))
        rep = self.activation(self.dropout(self.transform(rep)))
        return self.head(rep), rep


class BetaBlock(nn.Module):
    """ReNF-beta sub-forecaster k > 1 with latent skip connections.

    ``z = Proj(Drop(Norm(x))) + Drop(r_prev)``, ``t = body(z)`` (``body_layers``
    x Linear-Drop), ``r = GELU(t) + Drop(z) + Drop(r_prev)``, output ``head(r)``.
    """

    def __init__(
        self,
        width: int,
        horizon: int,
        d_ff: int,
        dropout: float,
        input_dropout: float,
        body_layers: int,
        block_norm: str,
    ) -> None:
        super().__init__()
        self.norm = VariateNorm() if block_norm == "instance" else nn.LayerNorm(width)
        self.input_dropout = nn.Dropout(input_dropout)
        self.projection = nn.Linear(width, d_ff)
        self.body = nn.Sequential(
            *[nn.Sequential(nn.Linear(d_ff, d_ff), nn.Dropout(dropout)) for _ in range(body_layers)]
        )
        self.activation = nn.GELU()
        self.head = nn.Linear(d_ff, horizon)
        self.drop_prev_in = nn.Dropout(dropout)
        self.drop_skip = nn.Dropout(dropout)
        self.drop_prev_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rep_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(self.input_dropout(self.norm(x))) + self.drop_prev_in(rep_prev)
        rep = self.activation(self.body(z)) + self.drop_skip(z) + self.drop_prev_out(rep_prev)
        return self.head(rep), rep


def _check_variant(variant: str, block_norm: str, body_layers: int) -> None:
    if variant not in {"alpha", "beta"}:
        raise ValueError("variant must be 'alpha' or 'beta'")
    if block_norm not in {"layer", "instance"}:
        raise ValueError("block_norm must be 'layer' or 'instance'")
    if variant == "alpha" and (block_norm != "layer" or body_layers != 1):
        raise ValueError("ReNF-alpha uses LayerNorm blocks with one transform layer")


def _check_blocks(num_blocks: int, pred_len: int) -> None:
    if num_blocks < 1 or pred_len % num_blocks != 0:
        raise ValueError(f"num_blocks={num_blocks} must divide pred_len={pred_len}")


class Model(nn.Module):
    """ReNF: Boosted Direct Output over N sub-forecasters of growing horizon.

    Sub-forecaster ``k`` maps the window concatenated (along time) with every
    previous sub-forecast to the first ``k * H / N`` steps; ``forward`` returns
    the last, full-horizon sub-forecast and ``sub_forecasts`` returns all of
    them for the block-wise supervision of Eq. (4).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        variant: str = "beta",
        num_blocks: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.5,
        body_layers: int = 1,
        block_norm: str = "layer",
        use_pe: bool = True,
        alpha_freq: float = 0.2,
    ) -> None:
        super().__init__()
        _check_variant(variant, block_norm, body_layers)
        _check_blocks(num_blocks, pred_len)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.variant = variant
        self.num_blocks = num_blocks
        self.alpha_freq = alpha_freq
        self.segment = pred_len // num_blocks
        # alpha: affine RevIN, detached sub-forecast inputs; beta: plain RevIN, latent skips.
        self.revin = RevIN(enc_in, affine=variant == "alpha")
        self.position = positional_encoding("zeros", True, seq_len, enc_in) if use_pe else None
        blocks: list[nn.Module] = [FirstBlock(seq_len, self.segment, d_ff, dropout)]
        for k in range(1, num_blocks):
            width = concat_width(seq_len, self.segment, k)
            horizon = self.segment * (k + 1)
            if variant == "alpha":
                input_dropout = min(0.3 * (k + 1), 0.6)
                blocks.append(AlphaBlock(width, horizon, d_ff, dropout, input_dropout, k == num_blocks - 1))
            else:
                input_dropout = max(0.1 * 0.8**k, 0.05)
                blocks.append(BetaBlock(width, horizon, d_ff, dropout, input_dropout, body_layers, block_norm))
        self.blocks = nn.ModuleList(blocks)

    def _normalized(self, x_enc: torch.Tensor) -> torch.Tensor:
        if x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(
                f"expected input [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        x = self.revin(x_enc, "norm")
        if self.position is not None:
            x = x + self.position
        return x

    def sub_forecasts(self, x_enc: torch.Tensor) -> list[torch.Tensor]:
        """All sub-forecasts ``[B, k * H / N, V]`` for k = 1..N, denormalized (Eq. 3)."""
        x = self._normalized(x_enc)
        window = x.transpose(1, 2)  # [B, V, T]; every projection acts on time
        outputs: list[torch.Tensor] = []
        block_input = window
        rep = None
        for block in self.blocks:
            if rep is None:
                forecast, rep = block(block_input)
            else:
                forecast, rep = block(block_input, rep)
            outputs.append(forecast)
            previous = torch.cat(outputs, dim=-1)
            if self.variant == "alpha":
                previous = previous.detach()
            block_input = torch.cat([window, previous], dim=-1)
        return [self.revin(y.transpose(1, 2), "denorm") for y in outputs]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.sub_forecasts(x_enc)[-1]

    def training_loss(self, forecasts: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Block-wise supervision (Eq. 4) with this model's frequency weight."""
        return block_supervision_loss(forecasts, target, self.alpha_freq)

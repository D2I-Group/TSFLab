"""ReCycle: Residual Cyclic Transformer (arXiv 2405.03429, IEEE CAI 2024).

Primary cycle compression (PCC) folds every univariate series into one token per
primary cycle (``cycle_len`` steps, a day in the paper); recent historic profiles
(RHP) average the last ``rhp_cycles`` cycles of the same day type (weekday,
Saturday, Sunday/holiday). An encoder-decoder Transformer encodes the historic
residuals (series minus RHP) with day metadata, decodes from the forecast RHP with
day metadata in one pass, and its output is added back to the forecast RHP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_DAY_TYPES = 3
META_FEATURES = 9  # weekday one-hot (7) + holiday today + holiday tomorrow


def day_types(weekdays: torch.Tensor) -> torch.Tensor:
    """Sec. VI-A-b categories: 0 = Monday-Friday, 1 = Saturday, 2 = Sunday (or holiday)."""
    return torch.where(
        weekdays == 6,
        torch.full_like(weekdays, 2),
        torch.where(weekdays == 5, torch.ones_like(weekdays), torch.zeros_like(weekdays)),
    )


def day_metadata(weekdays: torch.Tensor) -> torch.Tensor:
    """Per-cycle metadata ``[..., 9]``: weekday one-hot plus the two holiday flags (zero here)."""
    onehot = F.one_hot(weekdays, num_classes=7).float()
    return torch.cat([onehot, onehot.new_zeros(*onehot.shape[:-1], 2)], dim=-1)


def profile_table(cycles: torch.Tensor, types: torch.Tensor, rhp_cycles: int) -> torch.Tensor:
    """Recent historic profiles available after every cycle.

    ``cycles`` is ``[N, H, D]`` and ``types`` ``[N, H]``. Entry ``[n, l, t]`` is the
    mean of the last ``rhp_cycles`` cycles of type ``t`` among cycles ``0..l``
    (inclusive), or zeros when no such cycle exists yet. Returns ``[N, H, 3, D]``.
    """
    length = cycles.shape[1]
    index = torch.arange(length, device=cycles.device)
    causal = (index[None, :] <= index[:, None]).to(cycles.dtype)  # [l, j]
    profiles = []
    for day_type in range(NUM_DAY_TYPES):
        member = (types == day_type).to(cycles.dtype)  # [N, H]
        count = member.cumsum(dim=1)
        # Cycle j is among the last k of its type up to l when fewer than k later ones exist.
        recent = ((count[:, :, None] - count[:, None, :]) < rhp_cycles).to(cycles.dtype)
        weights = member[:, None, :] * causal[None] * recent  # [N, l, j]
        total = weights.sum(dim=-1, keepdim=True)
        profiles.append(torch.bmm(weights, cycles) / total.clamp_min(1.0))
    return torch.stack(profiles, dim=2)


def select_profiles(table: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Pick ``table[n, l, types[n, l]]`` from a ``[N, L, 3, D]`` table, giving ``[N, L, D]``."""
    index = types[:, :, None, None].expand(-1, -1, 1, table.shape[-1])
    return table.gather(2, index).squeeze(2)


class Model(nn.Module):
    """Channel-independent ReCycle with the vanilla Transformer of Sec. V-C."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle_len: int = 24,
        rhp_cycles: int = 3,
        rhp_mode: str = "last",
        d_model: int = 32,
        n_heads: int = 1,
        e_layers: int = 1,
        d_layers: int = 1,
        d_ff: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths, channels, widths, heads and layers must be positive")
        if seq_len % cycle_len or pred_len % cycle_len:
            raise ValueError("seq_len and pred_len must be whole multiples of cycle_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cycle_len = cycle_len
        self.history_cycles = seq_len // cycle_len
        self.forecast_cycles = pred_len // cycle_len
        self.rhp_cycles = rhp_cycles
        self.rhp_mode = rhp_mode
        token_dim = cycle_len + META_FEATURES
        # The official model widens d_model by the number of heads (per-head width).
        width = d_model * n_heads
        self.encoder_in = self._projection(token_dim, width, dropout)
        self.decoder_in = self._projection(token_dim, width, dropout)
        self.transformer = nn.Transformer(
            d_model=width,
            nhead=n_heads,
            num_encoder_layers=e_layers,
            num_decoder_layers=d_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.out = self._projection(width, cycle_len, dropout)

    @staticmethod
    def _projection(in_features: int, out_features: int, dropout: float) -> nn.Module:
        if in_features == out_features:
            return nn.Identity()
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, out_features))

    def cycle_weekdays(self, x_mark_enc, x_mark_dec, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Weekday (0 = Monday) of the first step of every historic and forecast cycle."""
        if x_mark_enc is None:
            history = torch.zeros(batch, self.history_cycles, dtype=torch.long, device=device)
        else:
            history = x_mark_enc[:, :: self.cycle_len, 3].round().long().remainder(7)
        if x_mark_dec is not None and x_mark_dec.ndim == 3 and x_mark_dec.shape[1] >= self.pred_len:
            future = x_mark_dec[:, -self.pred_len :: self.cycle_len, 3].round().long().remainder(7)
        else:
            # One primary cycle per day: the weekday advances by one per cycle.
            steps = torch.arange(1, self.forecast_cycles + 1, device=device)
            future = (history[:, -1:] + steps[None, :]).remainder(7)
        return history, future

    def profiles(
        self, cycles: torch.Tensor, history_types: torch.Tensor, forecast_types: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RHP of every historic and forecast cycle, ``[N, H, D]`` and ``[N, F, D]``.

        ``last`` (official default ``LooseTypeLastRHPDataset``): every cycle uses the
        profiles available after the last historic cycle. ``causal`` (Sec. V-B text):
        historic cycle ``l`` uses only cycles before it; forecast cycles use all history.
        """
        table = profile_table(cycles, history_types, self.rhp_cycles)
        last = table[:, -1:].expand(-1, forecast_types.shape[1], -1, -1)
        forecast = select_profiles(last, forecast_types)
        if self.rhp_mode == "last":
            history = select_profiles(table[:, -1:].expand(-1, cycles.shape[1], -1, -1), history_types)
        else:
            previous = torch.cat([torch.zeros_like(table[:, :1]), table[:, :-1]], dim=1)
            history = select_profiles(previous, history_types)
        return history, forecast

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in})")
        batch, channels = x_enc.shape[0], self.enc_in
        # PCC: [B, L, C] -> [B * C, H, D], one token per primary cycle and series.
        cycles = x_enc.permute(0, 2, 1).reshape(batch * channels, self.history_cycles, self.cycle_len)
        history_days, forecast_days = self.cycle_weekdays(x_mark_enc, x_mark_dec, batch, x_enc.device)
        history_days = history_days.repeat_interleave(channels, dim=0)
        forecast_days = forecast_days.repeat_interleave(channels, dim=0)
        history_rhp, forecast_rhp = self.profiles(cycles, day_types(history_days), day_types(forecast_days))
        # Encoder: historic residuals with day metadata; decoder: forecast RHP with day metadata.
        encoder_tokens = torch.cat([cycles - history_rhp, day_metadata(history_days)], dim=-1)
        decoder_tokens = torch.cat([forecast_rhp, day_metadata(forecast_days)], dim=-1)
        latent = self.transformer(self.encoder_in(encoder_tokens), self.decoder_in(decoder_tokens))
        forecast = self.out(latent) + forecast_rhp  # residual forecast added back to the RHP
        return forecast.reshape(batch, channels, self.pred_len).permute(0, 2, 1)

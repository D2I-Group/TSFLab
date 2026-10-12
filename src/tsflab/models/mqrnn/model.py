"""Clean-room MQ-RNN with the paper's global/local decoder decomposition.

The recurrent encoder consumes the target and historical temporal covariates.
The global decoder jointly maps its final state and every known-future
covariate to horizon-specific and horizon-agnostic contexts.  One local MLP,
shared over horizons, combines those contexts with the matching future
covariate before the repository's monotone quantile parameterization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.quantile_head import QuantileHead

_DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _check_sizes(seq_len: int, pred_len: int, enc_in: int, smallest_hidden: int) -> None:
    if seq_len < 1 or pred_len < 1 or enc_in < 1:
        raise ValueError("seq_len, pred_len, and enc_in must be positive")
    if smallest_hidden < 1:
        raise ValueError("all hidden dimensions and num_layers must be positive")


def check_quantile_levels(levels: list[float]) -> None:
    """Raise unless ``levels`` is a non-empty, strictly ascending list inside (0, 1)."""
    if not levels or any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("quantile_levels must be non-empty values in (0, 1)")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("quantile_levels must be strictly ascending")


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        hidden_size: int = 64,
        num_layers: int = 1,
        context_size: int = 32,
        decoder_hidden: int = 64,
        future_covariate_size: int = 6,
        dropout: float = 0.1,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        _check_sizes(seq_len, pred_len, enc_in, min(hidden_size, num_layers, context_size, decoder_hidden))
        if future_covariate_size < 0:
            raise ValueError("future_covariate_size must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.future_covariate_size = future_covariate_size
        self.context_size = context_size
        self.output_type = "quantile"
        levels = list(quantile_levels) if quantile_levels else _DEFAULT_LEVELS
        check_quantile_levels(levels)

        self.encoder = nn.LSTM(
            input_size=1 + future_covariate_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        global_output = (pred_len + 1) * context_size
        self.global_decoder = nn.Sequential(
            nn.Linear(hidden_size + pred_len * future_covariate_size, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, global_output),
        )
        self.local_decoder = nn.Sequential(
            nn.Linear(2 * context_size + future_covariate_size, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.quantile_head = QuantileHead(levels, in_features=decoder_hidden)

    def _check_marks(self, marks: torch.Tensor, batch: int) -> None:
        if marks.ndim != 3 or marks.shape[0] != batch:
            raise ValueError("MQRNN temporal covariates must be rank-3 and batch-aligned")
        if marks.shape[-1] != self.future_covariate_size:
            raise ValueError(
                "MQRNN expected temporal covariate width "
                f"{self.future_covariate_size}, got {marks.shape[-1]}"
            )

    def _covariates(
        self,
        marks: torch.Tensor | None,
        *,
        batch: int,
        length: int,
        reference: torch.Tensor,
        take_last: bool = False,
    ) -> torch.Tensor:
        if self.future_covariate_size == 0:
            return reference.new_zeros(batch, length, 0)
        if marks is None:
            return reference.new_zeros(batch, length, self.future_covariate_size)
        self._check_marks(marks, batch)
        if take_last:
            if marks.shape[1] < length:
                raise ValueError("future covariates do not cover the prediction horizon")
            return marks[:, -length:, :].to(dtype=reference.dtype)
        if marks.shape[1] != length:
            raise ValueError("historical covariates must match seq_len")
        return marks.to(dtype=reference.dtype)

    def decode_contexts(
        self, encoder_context: torch.Tensor, future_covariates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the paper's global context equation."""
        batch_series = encoder_context.shape[0]
        repeated_future = future_covariates.repeat_interleave(self.enc_in, dim=0)
        global_input = torch.cat(
            [encoder_context, repeated_future.reshape(batch_series, -1)], dim=-1
        )
        contexts = self.global_decoder(global_input)
        contexts = contexts.reshape(batch_series, self.pred_len + 1, self.context_size)
        return contexts[:, : self.pred_len], contexts[:, -1]

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"MQRNN expects input shaped (batch, {self.seq_len}, {self.enc_in})"
            )
        batch = x_enc.shape[0]
        historical_covariates = self._covariates(
            x_mark_enc,
            batch=batch,
            length=self.seq_len,
            reference=x_enc,
        )
        future_covariates = self._covariates(
            x_mark_dec,
            batch=batch,
            length=self.pred_len,
            reference=x_enc,
            take_last=True,
        )

        targets = x_enc.permute(0, 2, 1).reshape(batch * self.enc_in, self.seq_len, 1)
        historical = historical_covariates.repeat_interleave(self.enc_in, dim=0)
        encoder_input = torch.cat([targets, historical], dim=-1)
        _, (hidden, _) = self.encoder(encoder_input)
        horizon_context, common_context = self.decode_contexts(hidden[-1], future_covariates)

        local_future = future_covariates.repeat_interleave(self.enc_in, dim=0)
        common = common_context[:, None, :].expand(-1, self.pred_len, -1)
        local_input = torch.cat([horizon_context, common, local_future], dim=-1)
        local_features = self.local_decoder(local_input)
        quantiles = self.quantile_head(local_features)
        quantiles = quantiles.reshape(
            batch, self.enc_in, self.pred_len, quantiles.shape[-1]
        ).permute(0, 2, 1, 3)
        if self.features == "MS":
            quantiles = quantiles[:, :, -1:, :]
        return quantiles

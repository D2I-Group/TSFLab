"""Clean-room TiDE implementation from Das et al. (TMLR 2023).

The channel-independent path follows paper equations (3)-(4): dynamic
covariates are projected per time step, the history and all projected
covariates are flattened into a dense encoder, a dense decoder emits one
vector per horizon step, a temporal decoder consumes the corresponding future
covariate, and a global linear residual maps lookback to horizon.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    """The paper's dense-ReLU-dense-dropout block plus projected skip."""

    def __init__(self, input_width: int, hidden_width: int, output_width: int, dropout: float, bias: bool, normalize: bool = True) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_width, hidden_width, bias=bias)
        self.output = nn.Linear(hidden_width, output_width, bias=bias)
        self.skip = nn.Linear(input_width, output_width, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(output_width, elementwise_affine=True) if normalize else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nonlinear = self.output(torch.relu(self.hidden(x)))
        return self.normalization(self.skip(x) + self.dropout(nonlinear))


class Model(nn.Module):
    """Time-series Dense Encoder with optional temporal covariates."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        decoder_output_dim: int,
        time_feat_dim: int,
        dropout: float,
        bias: bool,
        feature_encode_dim: int,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, d_model, e_layers, d_layers, d_ff, decoder_output_dim, time_feat_dim, feature_encode_dim) <= 0:
            raise ValueError("all TiDE dimensions and layer counts must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.time_feat_dim = time_feat_dim
        self.feature_projection = ResidualBlock(time_feat_dim, d_model, feature_encode_dim, dropout, bias)
        encoder_input = seq_len + (seq_len + pred_len) * feature_encode_dim
        self.encoder_input = ResidualBlock(encoder_input, d_model, d_model, dropout, bias)
        self.encoder_blocks = nn.ModuleList(
            ResidualBlock(d_model, d_model, d_model, dropout, bias)
            for _ in range(e_layers - 1)
        )
        self.decoder_blocks = nn.ModuleList(
            ResidualBlock(d_model, d_model, d_model, dropout, bias)
            for _ in range(d_layers - 1)
        )
        self.dense_decoder = ResidualBlock(
            d_model, d_model, pred_len * decoder_output_dim, dropout, bias
        )
        # LayerNorm over a scalar would erase the nonlinear branch, so the
        # temporal decoder intentionally uses the paper's optional no-norm form.
        self.temporal_decoder = ResidualBlock(
            decoder_output_dim + feature_encode_dim, d_ff, 1, dropout, bias, normalize=False
        )
        self.global_residual = nn.Linear(seq_len, pred_len, bias=bias)
        self.decoder_output_dim = decoder_output_dim

    def _covariates(self, x: torch.Tensor, historical: torch.Tensor | None, future: torch.Tensor | None) -> torch.Tensor:
        batch = x.shape[0]
        if historical is None:
            historical = x.new_zeros(batch, self.seq_len, self.time_feat_dim)
        if future is None:
            future = x.new_zeros(batch, self.pred_len, self.time_feat_dim)
        else:
            future = future[:, -self.pred_len :]
        expected_historical = (batch, self.seq_len, self.time_feat_dim)
        expected_future = (batch, self.pred_len, self.time_feat_dim)
        if tuple(historical.shape) != expected_historical or tuple(future.shape) != expected_future:
            raise ValueError(
                f"TiDE marks must have shapes {expected_historical} and {expected_future}; "
                f"got {tuple(historical.shape)} and {tuple(future.shape)}"
            )
        return torch.cat((historical, future), dim=1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"expected [batch, {self.seq_len}, channels], got {tuple(x_enc.shape)}")
        covariates = self._covariates(x_enc, x_mark_enc, x_mark_dec)
        projected = self.feature_projection(covariates)
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        scale = x_enc.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        normalized = (x_enc - mean) / scale

        history = normalized.transpose(1, 2)
        chunk_size = self._channel_chunk_size(x_enc.shape[0], x_enc.shape[-1])
        channel_outputs = []
        for start in range(0, x_enc.shape[-1], chunk_size):
            chunk = history[:, start : start + chunk_size]
            if self.training and torch.is_grad_enabled() and x_enc.shape[-1] > chunk_size:
                forecast = checkpoint(self._forecast_channels, chunk, projected, use_reentrant=False)
            else:
                forecast = self._forecast_channels(chunk, projected)
            channel_outputs.append(forecast)
        output = torch.cat(channel_outputs, dim=1).transpose(1, 2)
        return output * scale[:, :1] + mean[:, :1]

    def _channel_chunk_size(self, batch: int, channels: int) -> int:
        # Bound the largest temporal hidden activation to about 128 MiB in fp32.
        # Checkpointing prevents retaining one such activation for every chunk.
        per_channel = batch * self.pred_len * self.temporal_decoder.hidden.out_features
        return max(1, min(channels, (32 * 1024 * 1024) // per_channel))

    def _forecast_channels(self, history: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = history.shape
        covariate_vector = projected.flatten(1).unsqueeze(1).expand(-1, channels, -1)
        encoded = self.encoder_input(torch.cat((history, covariate_vector), dim=-1))
        for block in self.encoder_blocks:
            encoded = block(encoded)
        decoded = encoded
        for block in self.decoder_blocks:
            decoded = block(decoded)
        decoded = self.dense_decoder(decoded).reshape(
            batch, channels, self.pred_len, self.decoder_output_dim
        )
        future_projected = projected[:, self.seq_len :].unsqueeze(1).expand(-1, channels, -1, -1)
        nonlinear = self.temporal_decoder(torch.cat((decoded, future_projected), dim=-1)).squeeze(-1)
        return nonlinear + self.global_residual(history)

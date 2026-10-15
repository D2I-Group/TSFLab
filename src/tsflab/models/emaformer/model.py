"""Local EMAformer: an inverted (variate-token) Transformer whose tokens carry
channel, phase, and joint channel-phase embeddings (Zhang et al., AAAI 2026).

Paper map: variate tokens (Eq. 4), channel embedding (Eq. 5), phase embedding
(Eq. 6), joint channel-phase embedding (Eq. 7), summed token input (Eq. 8),
post-norm encoder (Eqs. 9-12), MLP head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.marks import days_from_civil
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer
from tsflab.models._inputs import check_window


def _elapsed_minutes(stamp: torch.Tensor) -> torch.Tensor:
    """Minutes since 1970-01-01 00:00 from ``[..., 6]`` raw marks (int64)."""
    stamp = stamp.round().long()
    days = days_from_civil(stamp[..., 0], stamp[..., 1], stamp[..., 2])
    return days * 1440 + stamp[..., 4] * 60 + stamp[..., 5]


def forecast_phase(marks: torch.Tensor | None, cycle: int, batch: int, device: torch.device) -> torch.Tensor:
    """Phase ``t mod P`` of the first forecast step, shape ``[batch]`` (long).

    The absolute step index is the elapsed calendar time of the last history step
    divided by the sampling interval (difference of the last two marks), plus one.
    Missing or non-calendar marks give phase 0.
    """
    if marks is None or marks.ndim != 3 or marks.shape[-1] < 6 or marks.shape[1] < 2:
        return torch.zeros(batch, dtype=torch.long, device=device)
    last = _elapsed_minutes(marks[:, -1])
    step = (last - _elapsed_minutes(marks[:, -2])).clamp_min(1)
    return (torch.div(last, step, rounding_mode="floor") + 1).remainder(cycle).to(device)


class Model(nn.Module):
    """Encoder-only variate-token Transformer with EMAformer's auxiliary embeddings."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle: int = 24,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        output_proj_dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.cycle, self.d_model, self.use_norm = cycle, d_model, use_norm

        # Eq. (4): one token per variate from its whole lookback.
        self.token_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        # Eqs. (5)-(7): channel, phase, and joint channel-phase tables (Xavier-normal).
        self.channel_embedding = nn.Parameter(torch.empty(enc_in, d_model))
        self.phase_embedding = nn.Embedding(cycle, d_model)
        self.joint_embedding = nn.Embedding(cycle, enc_in * d_model)
        nn.init.xavier_normal_(self.channel_embedding)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        # Eqs. (9)-(12): post-norm self-attention encoder over variate tokens.
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        # Forecast head: d -> 2d -> 4d -> pred_len MLP.
        self.projector = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(2 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(4 * d_model, pred_len),
        )

    def armored_tokens(self, values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Eq. (8): ``Z0 = E_x + E_c + E_p + E_cp``, shape ``[B, C, d_model]``."""
        batch = values.shape[0]
        tokens = self.token_embedding(values, None)
        channel = self.channel_embedding.unsqueeze(0)
        phase_emb = self.phase_embedding(phase).unsqueeze(1)
        joint = self.joint_embedding(phase).view(batch, self.enc_in, self.d_model)
        return tokens + channel + phase_emb + joint

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        if self.use_norm:
            x_enc, means, stdev = standardize(x_enc)
        phase = forecast_phase(x_mark_enc, self.cycle, x_enc.shape[0], x_enc.device)
        tokens = self.armored_tokens(x_enc, phase)
        encoded, _ = self.encoder(tokens, attn_mask=None)
        # Official head reads the encoder output plus the armored input tokens.
        forecast = self.projector(encoded + tokens).transpose(1, 2)
        if self.use_norm:
            forecast = destandardize(forecast, means, stdev)
        return forecast

"""C-LoRA (Nie, Mei, Qin, Sun & Ma, CIKM 2024, arXiv:2407.17246).

Independent implementation of channel-aware low-rank adaptation as a plug-in on
the inverted (variate-token) template of paper Eq. (3). A shared token embedding
maps every channel's lookback to ``D = d_model - adaptation_dim`` features; each
channel owns a rank-``r`` factor that a shared low-rank map lifts to a ``D x d``
adapter (Eq. 5); the adapter projects the channel's own embedding to a ``d``-dim
identity-aware code (Eq. 6) that is concatenated to the shared embedding (Eq. 7)
before the channel-dependent backbone. Two backbones from the paper are provided:
``itransformer`` (attention across variate tokens) and ``rmlp`` (residual MLP).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer
from tsflab.models._inputs import check_window


class ChannelLowRankAdapter(nn.Module):
    """Eq. (5)-(6): per-channel low-rank adapters conditioned on the channel token.

    ``factors`` holds ``phi^(c)`` for every channel, stored ``[C, D, r]`` (the
    transpose of the paper's ``r x D``); ``lift`` is the shared ``W`` (``r -> d``,
    no bias). ``adapters()`` returns ``ReLU(phi^(c),T W)`` as ``[C, D, d]`` and
    ``forward`` returns ``z_c^T phi~^(c)`` for every channel, ``[B, C, d]``.
    """

    def __init__(self, channels: int, embed_dim: int, rank: int, adaptation_dim: int) -> None:
        super().__init__()
        self.factors = nn.Parameter(torch.empty(channels, embed_dim, rank))
        nn.init.xavier_uniform_(self.factors)
        self.lift = nn.Linear(rank, adaptation_dim, bias=False)

    def adapters(self) -> torch.Tensor:
        return F.relu(self.lift(self.factors))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [B, C, D]; adapters [C, D, d] -> [B, C, d]
        return torch.einsum("bcD,cDd->bcd", tokens, self.adapters())


class Model(nn.Module):
    """Channel-aware low-rank adaptation on an iTransformer or RMLP backbone."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        backbone: str = "itransformer",
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        rank: int = 16,
        adaptation_dim: int = 32,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.backbone = backbone
        embed_dim = d_model - adaptation_dim
        self.embed_dim = embed_dim

        # Eq. (3) TokenEmbedding R^T -> R^D, shared by all channels.
        self.embedding = DataEmbedding_inverted(seq_len, embed_dim, dropout=dropout)
        self.adapter = ChannelLowRankAdapter(enc_in, embed_dim, rank, adaptation_dim)
        if backbone == "itransformer":
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, attention_dropout=dropout),
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
            # Non-stationary-Transformer normalization: no affine parameters.
            self.revin = RevIN(enc_in, affine=False)
        else:
            self.temporal = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
            )
            self.revin = RevIN(enc_in, affine=True)
        self.projection = nn.Linear(d_model, pred_len)

    def adapted_tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        """Eq. (7): ``[TokenEmbedding(X) || Z_phi]`` as ``[B, C, d_model]``."""
        shared = self.embedding(normalized, None)
        return torch.cat((shared, self.adapter(shared)), dim=-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        tokens = self.adapted_tokens(normalized)
        if self.backbone == "itransformer":
            tokens, _ = self.encoder(tokens, attn_mask=None)
        else:
            tokens = tokens + self.temporal(tokens)
        forecast = self.projection(tokens).transpose(1, 2)
        return self.revin(forecast, "denorm")

"""Clean-room TimePerceiver latent encoder/query decoder."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, heads, hidden, dropout):
        super().__init__()
        self.context = nn.Linear(context_dim, query_dim)
        self.attention = nn.MultiheadAttention(query_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.feedforward = nn.Sequential(nn.Linear(query_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, query_dim))
        self.norm2 = nn.LayerNorm(query_dim)

    def forward(self, queries, context):
        projected = self.context(context)
        attended, weights = self.attention(queries, projected, projected, need_weights=True)
        hidden = self.norm1(queries + attended)
        return self.norm2(hidden + self.feedforward(hidden)), weights


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, label_len=0, features="M",
                 d_model=32, n_heads=2, patch_len=16, dropout=0.1,
                 num_latents=8, latent_dim=128, latent_d_ff=256,
                 num_latent_blocks=1, query_share=True):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len = patch_len
        self.patch_count = math.ceil(seq_len / patch_len)
        self.query_share = query_share
        self.revin = RevIN(enc_in)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.channel_embedding = nn.Parameter(torch.randn(enc_in, d_model) * 0.02)
        self.input_position = nn.Parameter(torch.randn(self.patch_count, d_model) * 0.02)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.encoder = CrossAttentionBlock(latent_dim, d_model, n_heads, latent_d_ff, dropout)
        encoder_layer = nn.TransformerEncoderLayer(latent_dim, n_heads, latent_d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.latent_blocks = nn.TransformerEncoder(encoder_layer, num_latent_blocks)
        query_count = pred_len if query_share else pred_len * enc_in
        self.target_queries = nn.Parameter(torch.randn(query_count, latent_dim) * 0.02)
        self.time_projection = nn.Linear(4, latent_dim)
        self.decoder = CrossAttentionBlock(latent_dim, latent_dim, n_heads, latent_d_ff, dropout)
        self.value_head = nn.Linear(latent_dim, enc_in if query_share else 1)
        self.last_encoder_attention = None
        self.last_decoder_attention = None

    @staticmethod
    def _time_features(marks, batch, steps, device, dtype):
        if marks is None:
            position = torch.linspace(0, 1, steps, device=device, dtype=dtype)
            return torch.stack((position, position.square(), torch.sin(2*torch.pi*position), torch.cos(2*torch.pi*position)), -1).expand(batch, -1, -1)
        selected = marks[:, -steps:]
        if selected.shape[-1] >= 6:
            return torch.stack((selected[..., 1]/12, selected[..., 2]/31, selected[..., 3]/7, selected[..., 4]/24), -1).to(dtype)
        selected = selected[..., :4].to(dtype)
        return F.pad(selected, (0, 4-selected.shape[-1]))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        normalized = self.revin(x_enc, "norm").transpose(1, 2)
        padded = F.pad(normalized, (0, self.patch_count*self.patch_len-self.seq_len))
        patches = self.patch_embedding(padded.unfold(-1, self.patch_len, self.patch_len))
        tokens = patches + self.channel_embedding[:, None, :] + self.input_position[None, :, :]
        tokens = tokens.flatten(1, 2)
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        latents, self.last_encoder_attention = self.encoder(latents, tokens)
        latents = self.latent_blocks(latents)
        marks = x_mark_dec if x_mark_dec is not None else None
        time = self.time_projection(self._time_features(marks, batch, self.pred_len, x_enc.device, x_enc.dtype))
        if self.query_share:
            queries = self.target_queries.unsqueeze(0) + time
            decoded, self.last_decoder_attention = self.decoder(queries, latents)
            output = self.value_head(decoded)
        else:
            queries = self.target_queries.reshape(self.pred_len, self.enc_in, -1)
            queries = queries.unsqueeze(0) + time.unsqueeze(2)
            decoded, self.last_decoder_attention = self.decoder(queries.flatten(1, 2), latents)
            output = self.value_head(decoded).reshape(batch, self.pred_len, self.enc_in)
        return self.revin(output, "denorm")

"""Clean-room Non-stationary Transformer for forecasting."""
from __future__ import annotations
import torch
from torch import nn

from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._inputs import check_window


class Projector(nn.Module):
    """Learn de-stationary factors from raw series and removed statistics."""
    def __init__(self, seq_len, channels, hidden_dims, output_dim):
        super().__init__()
        dims = [2 * channels, *hidden_dims, output_dim]
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.GELU())
        self.temporal_pool = nn.Linear(seq_len, 1)
        self.network = nn.Sequential(*layers)

    def forward(self, raw, statistic):
        pooled = self.temporal_pool(raw.transpose(1, 2)).squeeze(-1)
        return self.network(torch.cat((pooled, statistic.squeeze(1)), -1))


class DeStationaryAttention(nn.Module):
    """Attention logits ``softmax((tau QK^T + delta) / sqrt(d))``."""
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.scale = heads, (width // heads) ** -0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, tau, delta):
        batch, length, width = query.shape
        q = self.qkv(query)[..., :width].reshape(batch, length, self.heads, -1)
        kv = self.qkv(context)
        k = kv[..., width:2 * width].reshape(batch, context.shape[1], self.heads, -1)
        v = kv[..., 2 * width:].reshape(batch, context.shape[1], self.heads, -1)
        scores = torch.einsum("blhd,bshd->bhls", q, k) * tau[:, None, None, None]
        scores = scores + delta[:, None, None, :context.shape[1]]
        values = torch.einsum("bhls,bshd->blhd", self.dropout((scores * self.scale).softmax(-1)), v)
        return self.out(values.flatten(2))


class NSBlock(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.attention = DeStationaryAttention(width, heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, x, context, tau, delta):
        x = self.norm1(x + self.attention(x, context, tau, delta))
        return self.norm2(x + self.ffn(x))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, label_len, enc_in, features="M", d_model=128,
                 n_heads=8, e_layers=2, d_layers=1, d_ff=256, dropout=0.1,
                 p_hidden_dims=None, p_hidden_layers=2):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in, self.label_len = seq_len, pred_len, enc_in, label_len
        hidden = list(p_hidden_dims or [128, 128])[:p_hidden_layers]
        self.value_embedding = nn.Linear(enc_in, d_model)
        self.position = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.future_queries = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.encoder = nn.ModuleList([NSBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.decoder = nn.ModuleList([NSBlock(d_model, n_heads, d_ff, dropout) for _ in range(d_layers)])
        self.tau_learner = Projector(seq_len, enc_in, hidden, 1)
        self.delta_learner = Projector(seq_len, enc_in, hidden, seq_len)
        self.projection = nn.Linear(d_model, enc_in)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        raw = x_enc
        normalized, mean, std = standardize(raw, 1e-5, detach_mean=True, detach_std=True)
        tau = self.tau_learner(raw, std).clamp(-5, 5).exp().squeeze(-1)
        delta = self.delta_learner(raw, mean)
        encoded = self.value_embedding(normalized) + self.position
        for layer in self.encoder:
            encoded = layer(encoded, encoded, tau, delta)
        decoded = self.future_queries.expand(raw.shape[0], -1, -1)
        for layer in self.decoder:
            decoded = layer(decoded, encoded, tau, delta)
        return destandardize(self.projection(decoded), mean, std)

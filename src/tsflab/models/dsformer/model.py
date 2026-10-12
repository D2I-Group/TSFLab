"""Independent DSformer implementation from dual-sampling paper structure."""
from __future__ import annotations
import torch
from torch import nn
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def dual_sampling(x, samples):
    """Return piecewise and interval sampling views."""
    batch, channels, length = x.shape
    if length % samples:
        raise ValueError("seq_len must be divisible by num_samp")
    width = length // samples
    piecewise = x.reshape(batch, channels, samples, width)
    interval = x.reshape(batch, channels, width, samples).transpose(2, 3)
    return piecewise, interval


class TVABlock(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.temporal = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.variable = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.cross_gate = nn.Linear(2 * width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, view):
        batch, channels, samples, width = view.shape
        temporal = view.reshape(batch * channels, samples, width)
        temporal, _ = self.temporal(temporal, temporal, temporal, need_weights=False)
        variable = view.transpose(1, 2).reshape(batch * samples, channels, width)
        variable, _ = self.variable(variable, variable, variable, need_weights=False)
        variable = variable.reshape(batch, samples, channels, width).transpose(1, 2)
        temporal = temporal.reshape_as(view)
        gate = torch.sigmoid(self.cross_gate(torch.cat((temporal, variable), -1)))
        view = self.norm1(view + gate * temporal + (1.0 - gate) * variable)
        return self.norm2(view + self.ffn(view))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, label_len=0, features="M", num_layer=1,
                 muti_head=2, num_samp=2, dropout=0.15, if_node=True):
        super().__init__()
        if seq_len % num_samp:
            raise ValueError("seq_len must be divisible by num_samp")
        width = seq_len // num_samp
        if width % muti_head or seq_len % muti_head:
            raise ValueError("attention widths must be divisible by muti_head")
        self.seq_len, self.pred_len, self.enc_in, self.num_samp = seq_len, pred_len, enc_in, num_samp
        self.revin = RevIN(enc_in)
        self.piece_blocks = nn.ModuleList([TVABlock(width, muti_head, 2 * width, dropout) for _ in range(num_layer)])
        self.interval_blocks = nn.ModuleList([TVABlock(width, muti_head, 2 * width, dropout) for _ in range(num_layer)])
        self.node_mix = nn.Linear(2 * seq_len, seq_len) if if_node else None
        self.decoder_attention = nn.MultiheadAttention(seq_len, muti_head, dropout, batch_first=True)
        self.head = nn.Linear(seq_len, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm").transpose(1, 2)
        piece, interval = dual_sampling(normalized, self.num_samp)
        for block in self.piece_blocks:
            piece = block(piece)
        for block in self.interval_blocks:
            interval = block(interval)
        piece, interval = piece.flatten(2), interval.transpose(2, 3).flatten(2)
        merged = torch.cat((piece, interval), -1)
        merged = self.node_mix(merged) if self.node_mix is not None else 0.5 * (piece + interval)
        decoded, _ = self.decoder_attention(merged, merged, merged, need_weights=False)
        return self.revin(self.head(merged + decoded).transpose(1, 2), "denorm")

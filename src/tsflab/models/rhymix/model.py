"""RhyMix: adaptive cyclic and multi-scale convolution forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._inputs import check_window


class TMSBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.RMSNorm(channels)
        self.convs = nn.ModuleList(
            [nn.Conv1d(channels, channels, 3, padding=d, dilation=d, groups=channels) for d in (1, 2, 4, 8)]
        )
        hidden = max(8, min(64, channels // 8))
        self.attn = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = self.norm(x).transpose(1, 2)
        y = sum(F.gelu(conv(z)) for conv in self.convs) / 4
        gate = self.attn(x.mean(dim=1)).unsqueeze(1)
        return x + self.scale * self.dropout(y.transpose(1, 2)) * gate


class ForecastHeads(nn.Module):
    def __init__(self, channels: int, seq_len: int, pred_len: int, periods: tuple[int, ...]):
        super().__init__()
        self.seq_len, self.pred_len, self.periods = seq_len, pred_len, periods
        self.direct = nn.Linear(seq_len, pred_len)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)
        self.local = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        self.local_proj = nn.Linear(seq_len, pred_len)
        self.period_conv = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1))
        self.period_proj = nn.Linear(seq_len, pred_len)
        self.ma_kernel = 25

    def forward(self, z, gpast, gfut):
        # z: B,L,C. Temporal maps are shared across channels.
        x = z.transpose(1, 2)
        direct = self.direct(x)
        trend = F.avg_pool1d(F.pad(x, (12, 12), mode="reflect"), self.ma_kernel, stride=1)
        decomp = self.seasonal(x - trend) + self.trend(trend)
        local = self.local_proj(x + self.local(x))
        # Two image planes retain the channel and time axes.
        periodic_input = torch.stack((x, gpast.transpose(1, 2)), dim=1)
        periodic = self.period_proj(x + self.period_conv(periodic_input).squeeze(1)) + gfut.transpose(1, 2)
        return torch.stack((direct, decomp, local, periodic), dim=1).transpose(-1, -2)


class Model(nn.Module):
    """Paper-equation repair; CNN layout remains an explicit local adaptation."""

    def __init__(self, c_in: int, seq_len: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.c_in, self.seq_len, self.pred_len = c_in, seq_len, pred_len
        self.periods = (12, 24, 48, 168)
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))
        rank = min(64, max(8, c_in // 8))
        self.channel_norm = nn.LayerNorm(c_in)
        self.channel_mlp = nn.Sequential(nn.Linear(c_in, rank), nn.GELU(), nn.Dropout(dropout), nn.Linear(rank, c_in))
        self.channel_scale = nn.Parameter(torch.tensor(0.05))
        self.stats_norm = nn.LayerNorm(5)
        self.period_gate = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 4))
        self.cyclic = nn.ParameterList([nn.Parameter(torch.randn(p, c_in) * 0.02) for p in self.periods])
        self.cyclic_norm = nn.LayerNorm(c_in)
        self.blocks = nn.ModuleList([TMSBlock(c_in, dropout) for _ in range(2)])
        self.heads = ForecastHeads(c_in, seq_len, pred_len, self.periods)
        self.path_gate = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 4))
        self.hybrid_gate = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.c_in)
        mean = x_enc.mean(1, keepdim=True)
        std = x_enc.std(1, unbiased=False, keepdim=True)
        x = (x_enc - mean) / (std + 1e-5) * self.weight + self.bias
        feat = torch.stack(
            (x.mean(1), x.std(1, unbiased=False), x[:, -1], x[:, -1] - x[:, 0], (x[:, 1:] - x[:, :-1]).abs().mean(1)),
            dim=-1,
        )
        feat_n = self.stats_norm(feat)
        global_feat = feat_n.mean(1)
        x = x + self.channel_scale * self.channel_mlp(self.channel_norm(x))
        weights = torch.sigmoid(self.period_gate(feat_n))
        denom = weights.sum(-1).clamp_min(1e-6)
        gp, gf = 0, 0
        for k, (p, emb) in enumerate(zip(self.periods, self.cyclic)):
            ixp = torch.arange(self.seq_len, device=x.device) % p
            ixf = (self.seq_len + torch.arange(self.pred_len, device=x.device)) % p
            w = weights[..., k]
            gp = gp + emb[ixp].unsqueeze(0) * w.unsqueeze(1)
            gf = gf + emb[ixf].unsqueeze(0) * w.unsqueeze(1)
        gp, gf = self.cyclic_norm(gp / denom[:, None, :]), gf / denom[:, None, :]
        tcn = x
        for block in self.blocks:
            tcn = block(tcn)
        pathw = self.path_gate(global_feat).softmax(-1)
        hybrid = self.hybrid_gate(global_feat).softmax(-1)
        yc = (self.heads(x, gp, gf) * pathw[:, :, None, None]).sum(1)
        yt = (self.heads(tcn, gp, gf) * pathw[:, :, None, None]).sum(1)
        y = hybrid[:, 0, None, None] * yc + hybrid[:, 1, None, None] * yt
        y = (y - self.bias) / (self.weight + 1e-5) * (std + 1e-5) + mean
        return y

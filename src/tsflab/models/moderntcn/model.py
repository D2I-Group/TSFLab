"""Clean-room ModernTCN forecast implementation from the ICLR paper.

The backbone preserves the variable axis, uses large-kernel depthwise temporal
convolution, then two grouped pointwise ConvFFNs: first per variable and then
per feature. Optional small kernels form the training-time reparameterization
branch described in the paper.
"""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import SeriesDecomposition


class LargeKernelDepthwiseConv(nn.Module):
    def __init__(self, channels: int, large_kernel: int, small_kernel: int | None):
        super().__init__()
        self.large = nn.Conv1d(channels, channels, large_kernel,
                               padding=large_kernel // 2, groups=channels)
        self.small = None if small_kernel is None else nn.Conv1d(
            channels, channels, small_kernel, padding=small_kernel // 2, groups=channels
        )

    def forward(self, values):
        result = self.large(values)
        return result if self.small is None else result + self.small(values)


class ModernTCNBlock(nn.Module):
    """DWConv + variable-grouped ConvFFN + feature-grouped ConvFFN."""
    def __init__(self, variables: int, width: int, ratio: int, large_kernel: int,
                 small_kernel: int | None, dropout: float):
        super().__init__()
        channels, expanded = variables * width, variables * width * ratio
        self.variables, self.width = variables, width
        self.temporal = LargeKernelDepthwiseConv(channels, large_kernel, small_kernel)
        self.norm = nn.BatchNorm1d(width)
        self.variable_ffn = nn.Sequential(
            nn.Conv1d(channels, expanded, 1, groups=variables), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(expanded, channels, 1, groups=variables), nn.Dropout(dropout),
        )
        self.feature_ffn = nn.Sequential(
            nn.Conv1d(channels, expanded, 1, groups=width), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(expanded, channels, 1, groups=width), nn.Dropout(dropout),
        )

    def forward(self, hidden):
        batch, variables, width, patches = hidden.shape
        residual = hidden
        mixed = self.temporal(hidden.reshape(batch, variables * width, patches))
        mixed = self.norm(mixed.reshape(batch * variables, width, patches))
        mixed = self.variable_ffn(mixed.reshape(batch, variables * width, patches))
        mixed = mixed.reshape(batch, variables, width, patches).transpose(1, 2)
        mixed = self.feature_ffn(mixed.reshape(batch, width * variables, patches))
        return residual + mixed.reshape(batch, width, variables, patches).transpose(1, 2)


def _head_width(seq_len, dims, patch_size, patch_stride, downsample_ratio, use_multi_scale):
    patch_counts = []
    length = seq_len
    for stage, width in enumerate(dims):
        kernel = patch_size if stage == 0 else downsample_ratio
        stride = patch_stride if stage == 0 else downsample_ratio
        length = (length - kernel) // stride + 1
        if length < 1:
            raise ValueError("stage configuration collapses the temporal axis")
        patch_counts.append(width * length)
    return sum(patch_counts) if use_multi_scale else patch_counts[-1]


class ModernTCNBackbone(nn.Module):
    def __init__(self, seq_len, pred_len, variables, ffn_ratio, num_blocks,
                 large_size, small_size, dims, patch_size, patch_stride,
                 downsample_ratio, dropout, head_dropout, use_multi_scale,
                 revin, affine, subtract_last):
        super().__init__()
        self.seq_len, self.pred_len, self.variables = seq_len, pred_len, variables
        self.revin = RevIN(variables, affine=affine, subtract_last=subtract_last, enabled=revin)
        self.stems = nn.ModuleList()
        self.stages = nn.ModuleList()
        for stage, width in enumerate(dims):
            if stage == 0:
                self.stems.append(nn.Conv1d(1, width, patch_size, stride=patch_stride))
            else:
                self.stems.append(nn.Conv1d(dims[stage - 1], width, downsample_ratio,
                                            stride=downsample_ratio))
            self.stages.append(nn.Sequential(*[
                ModernTCNBlock(variables, width, ffn_ratio, large_size[stage],
                               small_size[stage], dropout)
                for _ in range(num_blocks[stage])
            ]))
        self.use_multi_scale = use_multi_scale
        head_width = _head_width(seq_len, dims, patch_size, patch_stride, downsample_ratio,
                                 use_multi_scale)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(head_width, pred_len))

    def forward_features(self, values, normalize=True):
        if normalize:
            values = self.revin(values.transpose(1, 2), "norm").transpose(1, 2)
        hidden = values.unsqueeze(2)
        scales = []
        for stem, blocks in zip(self.stems, self.stages, strict=True):
            batch, variables, width, length = hidden.shape
            if length < stem.kernel_size[0]:
                raise ValueError("stage input is shorter than its convolution kernel")
            hidden = stem(hidden.reshape(batch * variables, width, length))
            hidden = hidden.reshape(batch, variables, hidden.shape[1], hidden.shape[2])
            hidden = blocks(hidden)
            scales.append(hidden)
        return scales if self.use_multi_scale else [scales[-1]]

    def forward(self, values):
        normalized = self.revin(values, "norm").transpose(1, 2)
        scales = self.forward_features(normalized, normalize=False)
        features = torch.cat([scale.flatten(2) for scale in scales], -1)
        forecast = self.head(features).transpose(1, 2)
        return self.revin(forecast, "denorm")


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M", label_len=0,
                 ffn_ratio=1, num_blocks=(1,), large_size=(13,), small_size=(5,),
                 dims=(32,), patch_size=16, patch_stride=16,
                 downsample_ratio=2,
                 dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True,
                 affine=True, subtract_last=False,
                 decomposition=False, kernel_size=25):
        super().__init__()
        arrays = [list(value) for value in (num_blocks, large_size, small_size, dims)]
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.decomposition = decomposition
        kwargs = dict(seq_len=seq_len, pred_len=pred_len, variables=enc_in,
                      ffn_ratio=ffn_ratio, num_blocks=arrays[0], large_size=arrays[1],
                      small_size=arrays[2], dims=arrays[3], patch_size=patch_size,
                      patch_stride=patch_stride, downsample_ratio=downsample_ratio,
                      dropout=dropout, head_dropout=head_dropout,
                      use_multi_scale=use_multi_scale, revin=revin, affine=affine,
                      subtract_last=subtract_last)
        if decomposition:
            self.decompose = SeriesDecomposition(kernel_size)
            self.seasonal = ModernTCNBackbone(**kwargs)
            self.trend = ModernTCNBackbone(**kwargs)
        else:
            self.backbone = ModernTCNBackbone(**kwargs)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        if not self.decomposition:
            return self.backbone(x_enc)
        seasonal, trend = self.decompose(x_enc)
        return self.seasonal(seasonal) + self.trend(trend)

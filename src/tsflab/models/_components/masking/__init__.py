"""Attention masks used by transformer variants."""

from __future__ import annotations

import math

import numpy as np
import torch


class TriangularCausalMask:
    def __init__(self, batch_size, length, device="cpu"):
        mask_shape = [batch_size, 1, length, length]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, batch_size, num_heads, length, index, scores, device="cpu"):
        mask = torch.ones(length, scores.shape[-1], dtype=torch.bool)
        mask = mask.to(device).triu(1)
        mask_expanded = mask[None, None, :].expand(batch_size, num_heads, length, scores.shape[-1])
        indicator = mask_expanded[
            torch.arange(batch_size)[:, None, None],
            torch.arange(num_heads)[None, :, None],
            index,
            :,
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class LocalMask:
    def __init__(self, batch_size, length, series_length, device="cpu"):
        mask_shape = [batch_size, 1, length, series_length]
        with torch.no_grad():
            local_len = math.ceil(np.log2(length))
            mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
            mask1 = mask1.to(device)
            mask2 = ~torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=-local_len).to(device)
            self._mask = mask1 + mask2

    @property
    def mask(self):
        return self._mask

"""Frequency-band decomposition mixture of experts.

Splits the ``rfft`` spectrum of a normalized ``(batch, channels, time)`` series
into ``expert_num`` contiguous frequency bands using learned, sorted band
boundaries, then combines the (masked) per-band spectra back into a single
signal with input-conditioned softmax gating weights. Each frequency band acts
as one "expert"; the gate decides, per input, how much each band contributes
to the reconstructed signal. This is a paper-neutral building block for
frequency-domain mixture-of-experts forecasters.
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyBandMixtureOfExperts(nn.Module):
    """Decompose a series into frequency bands and gate their mixture.

    Args:
        expert_num: Number of bands/experts.
        seq_len: Exact input length.
        learnable_boundaries: ``False`` (default, the behavior of the official
            code, whose integer cast blocks every gradient) stores the raw
            boundaries as a fixed buffer ``band_boundaries``. ``True`` makes
            ``band_boundaries`` an ``nn.Parameter`` trained through a
            straight-through estimator: the forward masks stay the exact hard
            0/1 band masks, the backward pass uses a sigmoid-edged soft mask.
        boundary_temperature: Soft-edge width in frequency bins for the
            straight-through gradient (only used when learnable).
    """

    def __init__(
        self,
        expert_num: int,
        seq_len: int,
        learnable_boundaries: bool = False,
        boundary_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if expert_num < 1:
            raise ValueError("expert_num must be positive")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.expert_num = expert_num
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1
        if boundary_temperature <= 0:
            raise ValueError("boundary_temperature must be positive")
        self.learnable_boundaries = bool(learnable_boundaries)
        self.boundary_temperature = float(boundary_temperature)
        initial = torch.rand(max(expert_num - 1, 0))
        if self.learnable_boundaries:
            self.band_boundaries = nn.Parameter(initial)
        else:
            # Same state-dict key as the former Parameter, so old checkpoints load.
            self.register_buffer("band_boundaries", initial)
        self.gating_network = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.ReLU(),
            nn.Linear(self.freq_len, expert_num),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize, band-decompose, and gate-combine ``x`` shaped ``[B, C, T]``.

        Returns ``(combined_output, boundaries, gating_scores)`` where
        ``combined_output`` is restored to the input's instance scale.
        """
        if x.ndim != 3 or x.shape[-1] != self.seq_len:
            raise ValueError(f"expected [batch, channels, {self.seq_len}], got {tuple(x.shape)}")
        x_mean = torch.mean(x, dim=2, keepdim=True)
        centered = x - x_mean
        x_var = torch.var(centered, dim=2, keepdim=True) + 1e-5
        normalized = centered / torch.sqrt(x_var)

        freq_x = torch.fft.rfft(normalized, dim=-1)
        total_freq_size = freq_x.size(-1)

        if self.expert_num > 1:
            boundaries = torch.sort(torch.sigmoid(self.band_boundaries)).values
        else:
            boundaries = self.band_boundaries
        boundaries = torch.cat(
            [
                torch.tensor([0.0], device=boundaries.device, dtype=boundaries.dtype),
                boundaries,
                torch.tensor([1.0], device=boundaries.device, dtype=boundaries.dtype),
            ]
        )
        edges = boundaries * total_freq_size
        indices = edges.detach().long()
        indices[-1] = total_freq_size

        bins = torch.arange(total_freq_size, device=freq_x.device)
        starts, ends = indices[:-1].unsqueeze(-1), indices[1:].unsqueeze(-1)
        hard = ((bins >= starts) & (bins < ends)).to(freq_x.real.dtype)  # [E, F]
        if self.learnable_boundaries and self.expert_num > 1:
            centers = bins.to(edges.dtype) + 0.5
            cdf = torch.sigmoid((centers - edges.unsqueeze(-1)) / self.boundary_temperature)
            soft = cdf[:-1] - cdf[1:]
            band_mask = hard + (soft - soft.detach()).to(hard.dtype)
        else:
            band_mask = hard
        # [B, C, F, E]
        components = freq_x.unsqueeze(-1) * band_mask.t().to(freq_x.real.dtype)

        gating_input = freq_x.abs().mean(dim=1)
        gating_scores = torch.softmax(self.gating_network(gating_input), dim=-1)

        combined_freq = torch.sum(components * gating_scores.unsqueeze(1).unsqueeze(2), dim=-1)
        combined_output = torch.fft.irfft(combined_freq, n=self.seq_len)
        combined_output = combined_output * torch.sqrt(x_var) + x_mean
        return combined_output, boundaries, gating_scores


__all__ = ["FrequencyBandMixtureOfExperts"]

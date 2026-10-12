"""Paper-driven local implementation of LIFT with a DLinear backbone.

LIFT refines a backbone's forecast with leading indicators: a non-parametric Lead
Estimator finds, per target variate, the K variates whose (lagged) cross-correlation
is largest (Eq. 2-4); the leaders are shifted to align with the horizon (Eq. 5-6); and
an adaptive frequency mixer, with state-dependent filters (Eq. 7, 9), refines the
normalized forecast in the frequency domain (Eq. 8, 10-12).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._inputs import check_window


class ComplexLinear(nn.Module):
    """Complex affine map on the last axis, with real and imaginary parameters."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias_re = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))
        self.bias_im = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.real, x.imag
        out_re = real @ self.weight_re.T - imag @ self.weight_im.T + self.bias_re
        out_im = real @ self.weight_im.T + imag @ self.weight_re.T + self.bias_im
        return torch.complex(out_re, out_im)


class Model(nn.Module):
    """DLinear backbone followed by the LIFT lead-aware refiner."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        leader_num: int = 4,
        state_num: int = 8,
        temperature: float = 1.0,
        kernel_size: int = 25,
        lead_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if seq_len < 4:
            raise ValueError("seq_len must be at least 4")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.leaders = min(leader_num, enc_in)
        self.states = state_num
        self.lead_chunk_size = lead_chunk_size
        freq = pred_len // 2 + 1
        self.freq = freq
        self.backbone = DLinearBackbone(enc_in, seq_len, pred_len, kernel_size=kernel_size)
        self.temperature = nn.Parameter(torch.tensor([float(temperature)]))
        # Eq. (7): P = softmax(P0 + f_state(X)), P0 per variate
        self.state_prior = nn.Parameter(torch.empty(enc_in, state_num))
        self.state_bias = nn.Parameter(torch.empty(state_num))
        self.state_classifier = nn.Linear(seq_len, state_num, bias=False)
        # Eq. (9): one linear filter factory f_n per state, K -> (2K+1) * freq
        out = (2 * self.leaders + 1) * freq
        self.factory_weight = nn.Parameter(torch.empty(state_num, self.leaders, out))
        self.factory_bias = nn.Parameter(torch.empty(state_num, out))
        # Eq. (11): complex linear g over [V, sum U, sum Delta]
        self.mix = ComplexLinear(3 * freq, freq)
        nn.init.uniform_(self.state_prior, -1 / math.sqrt(state_num), 1 / math.sqrt(state_num))
        nn.init.uniform_(self.state_bias, -1 / math.sqrt(seq_len), 1 / math.sqrt(seq_len))
        bound = 1 / math.sqrt(self.leaders)
        nn.init.uniform_(self.factory_weight, -bound, bound)
        nn.init.uniform_(self.factory_bias, -bound, bound)

    # -- Sec. 3.2: Lead Estimator ------------------------------------------------------
    @torch.no_grad()
    def estimate_leaders(self, normalized: torch.Tensor):
        """Return leader ids, leading steps, and signed correlations, each ``(B, C, K)``.

        ``corr[b, j, i, tau] = mean_t x_j[t + tau] x_i[t]`` (circular, Eq. 2); only local
        maxima over ``tau`` are candidates and ``tau = 0`` is excluded (Eq. 3-4).
        """
        length = normalized.shape[-1]
        spectrum = torch.fft.rfft(normalized, dim=-1)
        strengths, lags, signeds = [], [], []
        # chunk over the first variate axis (official code uses 32) so that the full
        # [B, C, C, L] correlation tensor is never materialised; results are identical
        for start in range(0, spectrum.shape[1], self.lead_chunk_size):
            block = spectrum[:, start : start + self.lead_chunk_size]
            corr = torch.fft.irfft(
                block.unsqueeze(2) * spectrum.conj().unsqueeze(1), n=length, dim=-1
            ) / length
            magnitude = corr.abs()
            peak = (magnitude[..., 1:-1] >= magnitude[..., :-2]) & (magnitude[..., 1:-1] >= magnitude[..., 2:])
            corr = corr[..., 1:-1] * peak
            strength, lag = corr.abs().max(-1)  # (B, chunk, C)
            strengths.append(strength)
            lags.append(lag)
            signeds.append(corr.gather(-1, lag.unsqueeze(-1)).squeeze(-1))
        strength, lag, signed = torch.cat(strengths, 1), torch.cat(lags, 1), torch.cat(signeds, 1)
        _, leaders = strength.topk(self.leaders, dim=-1)
        return leaders, lag.gather(2, leaders) + 1, signed.gather(2, leaders)

    # -- Eq. (5)-(6): target-oriented shifts -----------------------------------------------
    @torch.no_grad()
    def shift_leaders(self, normalized, forecast, leaders, shift, corr):
        """Align leaders to each target's horizon: ``(B, C, K, H)``."""
        batch, channels, length = normalized.shape
        horizon = forecast.shape[-1]
        sequence = torch.cat([normalized, forecast], dim=-1)  # observed, then predicted
        rows = sequence.gather(1, leaders.reshape(batch, -1, 1).expand(-1, -1, length + horizon))
        steps = torch.arange(length, length + horizon, device=sequence.device).view(1, 1, -1)
        steps = steps - shift.reshape(batch, -1, 1)
        aligned = rows.gather(-1, steps).view(batch, channels, self.leaders, horizon)
        return aligned * torch.sign(corr).unsqueeze(-1)

    def lead_filters(self, lookback: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Eq. (7), (9): filters ``(B, C, 2K+1, freq)`` from correlations and states."""
        batch, channels = corr.shape[:2]
        # constant-one logit competes with the |corr| logits (official scaling of R)
        logits = torch.cat([torch.ones_like(corr[..., :1]), corr.abs()], dim=-1)
        strength = torch.softmax(logits / self.temperature, dim=-1)[..., 1:]
        state = torch.softmax(
            self.state_bias + self.state_prior + self.state_classifier(lookback), dim=-1
        )  # (B, C, N)
        per_state = torch.einsum("bck,nkf->bcnf", strength, self.factory_weight)
        filters = torch.einsum("bcn,bcnf->bcf", state, per_state) + state @ self.factory_bias
        return filters.view(batch, channels, 2 * self.leaders + 1, self.freq)

    def refine(self, lookback: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        """Refine ``forecast (B, C, H)`` with the leaders of ``lookback (B, C, L)``."""
        mean = lookback.mean(-1, keepdim=True)
        centered = lookback - mean
        scale = (centered.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt()
        normalized = centered / scale
        predicted = (forecast - mean) / scale
        leaders, shift, corr = self.estimate_leaders(normalized)
        shifted = self.shift_leaders(normalized, predicted, leaders, shift, corr)
        filters = self.lead_filters(lookback, corr)
        k = self.leaders
        v = torch.fft.rfft(predicted, dim=-1)  # (B, C, F)
        u = torch.fft.rfft(shifted, dim=-1) * filters[:, :, :k, :]  # r_U ⊙ U
        delta = (u - v.unsqueeze(2)) * filters[:, :, k:2 * k, :]
        v = v * filters[:, :, 2 * k, :]
        mixed = self.mix(torch.cat([u.sum(2), delta.sum(2), v], dim=-1))
        refined = predicted + torch.fft.irfft(mixed, n=self.pred_len, dim=-1)
        return refined * scale + mean

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        forecast = self.backbone(x_enc)
        refined = self.refine(x_enc.transpose(1, 2), forecast.transpose(1, 2))
        return refined.transpose(1, 2)

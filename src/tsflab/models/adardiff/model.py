"""Causal adaptive differencing, residual pretraining and finite reconstruction."""
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._inputs import check_window


def polynomial_product(a, b, length):
    """Linear convolution with sufficient zero padding, truncated causally."""
    nfft = 1 << (a.shape[-1] + b.shape[-1] - 2).bit_length()
    return torch.fft.irfft(torch.fft.rfft(a, n=nfft) * torch.fft.rfft(b, n=nfft), n=nfft)[..., :length]


def causal_inverse_kernel(delta, horizon):
    """Finite formal-series inverse via Newton doubling; no circular reciprocal."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    g = torch.ones_like(delta[..., :1])
    while g.shape[-1] < horizon:
        length = min(2 * g.shape[-1], horizon)
        a = torch.cat([torch.ones_like(delta[..., :1]), -delta[..., :length-1]], dim=-1)
        ag = polynomial_product(a, g, length)
        ag = F.pad(ag, (0, length-ag.shape[-1]))
        identity_twice = F.pad(torch.full_like(g[..., :1], 2.), (0, length-1))
        g = polynomial_product(g, identity_twice-ag, length)
    return g


class Forecaster(nn.Module):
    def __init__(self, channels: int, lookback: int, horizon: int, p: int, enabled: bool):
        super().__init__()
        self.channels, self.lookback, self.horizon = channels, lookback, horizon
        self.p, self.enabled = p, enabled
        self.linear = ChannelWiseLinear(lookback - 1 if enabled else lookback, horizon, channels)
        if enabled:
            self.delta = nn.Parameter(torch.zeros(channels, p))
        # FFT lengths are chosen so circular wrap cannot affect the first H coefficients.
        self.nfft = 1 << (2 * horizon + p - 1).bit_length()

    def difference(self, x):  # [B,T,C] -> residuals at t=2..T
        return self._difference_slice(x, 1, x.shape[1])

    def _difference_slice(self, x, start, end):
        b, _, c = x.shape
        left = x[:, :1, :].expand(-1, self.p, -1)
        padded = torch.cat([left, x], dim=1).transpose(1, 2)
        kernel = torch.cat([-self.delta.flip(-1), torch.ones(c, 1, device=x.device, dtype=x.dtype)], dim=1).unsqueeze(1)
        result = F.conv1d(padded, kernel, groups=c).transpose(1, 2)
        return result[:, start:end, :]

    def forward(self, x, target=None):
        if not self.enabled:
            return self.linear(x.transpose(1, 2)).transpose(1, 2)
        z = self.difference(x)
        zhat = self.linear(z.transpose(1, 2)).transpose(1, 2)
        if target is None:
            return zhat
        # Initial-condition term c_k = sum_{j>=k} delta_j x_{L+k-j}.
        hist = x[:, -self.p:, :].flip(1).transpose(1, 2).reshape(-1, 1, self.p)
        signal = self.delta.unsqueeze(0).expand(x.shape[0], -1, -1).reshape(1, -1, self.p)
        signal = F.pad(signal, (0, self.horizon - 1))
        c = F.conv1d(signal, hist, groups=x.shape[0] * self.channels).reshape(x.shape[0], self.channels, self.horizon).transpose(1, 2)
        # h is the inverse generating function (1 - sum_j delta_j z^j)^-1.
        h = causal_inverse_kernel(self.delta, self.horizon)
        v = zhat + c
        # Causal convolution independently for each batch/channel.
        vf = torch.fft.rfft(v.transpose(1, 2), n=self.nfft, dim=-1)
        hf = torch.fft.rfft(h, n=self.nfft, dim=-1)
        return torch.fft.irfft(vf * hf.unsqueeze(0), n=self.nfft, dim=-1)[..., :self.horizon].transpose(1, 2)


class NativeForecaster(Forecaster):
    """Native forecasting interface; residual supervision owns a separate stage."""
    def __init__(self, channels, lookback, horizon, p, enabled=True,
                 phase1_epochs=30, phase1_lr=.001, phase1_patience=3):
        super().__init__(channels, lookback, horizon, p, enabled)
        self.phase1_epochs = phase1_epochs
        self.phase1_lr = phase1_lr
        self.phase1_patience = phase1_patience
        self.phase1_history = []

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.lookback, self.channels)
        # No target values are consumed by reconstruction.
        return super().forward(x_enc, target=True)

    def residual_pair(self, x, target):
        prediction = super().forward(x)
        expected = self._difference_slice(torch.cat([x, target], dim=1),
                                          x.shape[1], x.shape[1]+self.horizon)
        return prediction, expected

    def _phase1_pass(self, loader, device, optimizer=None):
        """Mean residual MSE over one pass of ``loader``; updates parameters when given an optimizer."""
        training = optimizer is not None
        self.train(training)
        total, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for batch in loader:
                x = batch[0].float().to(device)
                target = batch[1].float().to(device)[:, -self.horizon:, :self.channels]
                prediction, expected = self.residual_pair(x, target)
                loss = F.mse_loss(prediction, expected)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite AdaRDiff residual loss")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total += loss.detach().item() * target.numel()
                count += target.numel()
        if not count:
            raise ValueError(f"Empty AdaRDiff {'train' if training else 'val'} residual loader")
        return total / count

    def _snapshot(self):
        return {key: value.detach().clone() for key, value in self.state_dict().items()}

    def pretrain(self, train_loader, device, *, val_loader):
        """Phase1 residual loss, train updates only; restore best validation state.

        Phase1 residual validation and patience are explicit protocol choices;
        the paper does not supply the exact early-stopping implementation.
        Native phase2 constructs a fresh optimizer, also a disclosed choice.
        """
        if not self.enabled or not self.phase1_epochs:
            return
        if val_loader is None:
            raise ValueError("Residual pretraining requires a validation loader")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.phase1_lr)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.phase1_epochs):
            totals = {"train": self._phase1_pass(train_loader, device, optimizer)}
            totals["val"] = self._phase1_pass(val_loader, device)
            self.phase1_history.append(dict(epoch=epoch + 1, **totals))
            if totals["val"] < best:
                best, stale, best_state = totals["val"], 0, self._snapshot()
            else:
                stale += 1
            if stale >= self.phase1_patience:
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.train()


class Model(NativeForecaster):
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

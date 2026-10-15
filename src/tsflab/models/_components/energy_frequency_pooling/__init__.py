"""Energy-weighted stochastic pooling across a token axis in the frequency domain.

Given a complex spectrum with a "token" axis (channels, variables, or any other
group of parallel series) and a frequency axis, this operator picks, for every
frequency bin independently, one token's spectral value using a softmax
distribution over per-bin spectral energy across tokens. The picked value is
then broadcast back to every token position, producing a single shared
"key-frequency" representation that every token can be fused with.

During training the pick is a genuine stochastic sample (``torch.multinomial``)
so gradients still flow through the sampled complex value while the routing
decision itself is not required to be differentiable. During evaluation the
pick is the arg-max of the same distribution, which keeps inference
deterministic (a documented train/eval difference; the paper only describes
the training-time stochastic behaviour).
"""

from __future__ import annotations

import torch
from torch import nn


class EnergyBasedFrequencyPooling(nn.Module):
    """Softmax-energy-weighted pooling of a complex spectrum across tokens."""

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Pool ``spectrum`` (complex, shape ``[batch, tokens, freq]``) across tokens.

        Returns a tensor of the same shape where every token position along
        dim 1 holds the same, energy-weighted-pooled complex value for a given
        frequency bin.
        """
        if not torch.is_complex(spectrum):
            raise ValueError("spectrum must be a complex tensor")
        if spectrum.ndim != 3:
            raise ValueError("spectrum must have shape [batch, tokens, freq]")
        batch, tokens, freq = spectrum.shape
        energy = spectrum.abs().pow(2)
        ratio = torch.softmax(energy, dim=1)
        flat_ratio = ratio.permute(0, 2, 1).reshape(batch * freq, tokens)
        if self.training:
            indices = torch.multinomial(flat_ratio, 1)
        else:
            indices = flat_ratio.argmax(dim=-1, keepdim=True)
        indices = indices.view(batch, freq, 1).permute(0, 2, 1)
        pooled = torch.gather(spectrum, 1, indices)
        return pooled.expand(-1, tokens, -1)


__all__ = ["EnergyBasedFrequencyPooling"]

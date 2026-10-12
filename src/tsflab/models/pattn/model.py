"""Independent PAttn implementation from Tan et al. (NeurIPS 2024).

Figure 4 and appendix D.3 specify channel-independent instance normalization,
overlapping patches, patch projection, one self-attention layer, and a linear
forecast projection. The paper explicitly distinguishes PAttn from PatchTST by
omitting positional embeddings and the Transformer feed-forward sublayer.
"""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from tsflab.models._components.instance_norm import destandardize, standardize


class Model(nn.Module):
    """Patch-and-attention forecasting baseline."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        features: str = "M",
        d_model: int = 128,
        n_heads: int = 8,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.patch_len = patch_len
        self.stride = stride
        self.patch_count = (seq_len + stride - patch_len) // stride + 1
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.forecast_projection = nn.Linear(self.patch_count * d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"expected [batch, {self.seq_len}, channels], got {tuple(x_enc.shape)}")
        normalized, mean, scale = standardize(x_enc)
        channels_first = normalized.transpose(1, 2)
        padded = F.pad(channels_first, (0, self.stride), mode="replicate")
        patches = padded.unfold(-1, self.patch_len, self.stride)
        if patches.shape[2] != self.patch_count:
            raise RuntimeError("internal patch-count calculation disagrees with unfold")
        batch, channels, patch_count, _ = patches.shape
        tokens = self.patch_projection(patches).reshape(batch * channels, patch_count, -1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + attended)
        forecast = self.forecast_projection(tokens.flatten(1))
        forecast = forecast.reshape(batch, channels, self.pred_len).transpose(1, 2)
        forecast = destandardize(forecast, mean, scale)
        return forecast[:, :, -1:] if self.features == "MS" else forecast

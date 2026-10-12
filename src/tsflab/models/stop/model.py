"""Clean-room STOP implementation from the ICML 2025 equations."""
from __future__ import annotations
import torch
from torch import nn
from tsflab.models._components.marks import to_calendar_spatiotemporal
from tsflab.models._components.series_decomposition import SeriesDecomposition


class ChannelMixer(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                                  nn.Linear(4 * width, width)) for _ in range(layers))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class CentralizedInteraction(nn.Module):
    """Equations (8)--(9): node↔ConAU low-rank attention, never node↔node."""
    def __init__(self, width: int, units: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.context = nn.Parameter(torch.randn(units, width) * 0.02)
        self.query = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.last_aggregation: torch.Tensor | None = None
        self.last_diffusion: torch.Tensor | None = None

    def forward(self, nodes: torch.Tensor, perturbation: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = nodes.shape
        q = self.query(nodes).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.context.reshape(1, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum("bhnd,bhkd->bhnk", q, k) / self.head_dim ** 0.5
        diffusion = logits.softmax(-1)
        aggregation_logits = logits.transpose(-1, -2)
        if perturbation is not None:
            aggregation_logits = aggregation_logits + perturbation[:, None, None, :]
        aggregation = aggregation_logits.softmax(-1)
        context_values = torch.einsum("bhkn,bhnd->bhkd", aggregation, q)
        result = torch.einsum("bhnk,bhkd->bhnd", diffusion, context_values)
        self.last_aggregation, self.last_diffusion = aggregation, diffusion
        return self.output(result.transpose(1, 2).reshape(b, n, d))


class Model(nn.Module):
    """Spatio-Temporal OOD Processor with centralized context units."""
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, model_dim: int = 16,
                 prompt_dim: int = 16, num_layer: int = 2, hid_dim: int = 64,
                 tod_size: int = 24, kernel_size: int = 3, core: int = 4,
                 head: int = 4) -> None:
        super().__init__()
        if min(seq_len, pred_len) <= 0:
            raise ValueError("STOP dimensions must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.decomposition = SeriesDecomposition(kernel_size)
        self.long_encoder = nn.Linear(seq_len, model_dim)
        self.short_encoder = nn.Linear(seq_len, model_dim)
        self.tod_prompt = nn.Embedding(tod_size, prompt_dim)
        self.dow_prompt = nn.Embedding(7, prompt_dim)
        width = model_dim + 2 * prompt_dim
        self.temporal_mixer = ChannelMixer(width, num_layer)
        self.central = CentralizedInteraction(width, core, head)
        self.refine = nn.Sequential(nn.Linear(2 * width, hid_dim), nn.GELU(), nn.Linear(hid_dim, width))
        self.spatial_mixer = ChannelMixer(width, num_layer)
        self.temporal_head = nn.Linear(width, pred_len)
        self.spatial_head = nn.Linear(width, pred_len)

    def _representation(self, x: torch.Tensor, marks: torch.Tensor | None) -> torch.Tensor:
        residual, trend = self.decomposition(x)
        values = self.long_encoder(trend.transpose(1, 2)) + self.short_encoder(residual.transpose(1, 2))
        if marks is None:
            tod = torch.zeros(x.shape[0], self.seq_len, dtype=torch.long, device=x.device)
            dow = torch.zeros_like(tod)
        else:
            calendar = to_calendar_spatiotemporal(x, marks)
            tod = (calendar[:, :, 0, 1] * self.tod_prompt.num_embeddings).long() % self.tod_prompt.num_embeddings
            dow = (calendar[:, :, 0, 2] * 7).long() % 7
        prompt = torch.cat((self.tod_prompt(tod).mean(1), self.dow_prompt(dow).mean(1)), -1)
        return torch.cat((values, prompt[:, None].expand(-1, self.enc_in, -1)), -1)

    def _forecast(self, representation: torch.Tensor,
                  perturbation: torch.Tensor | None = None) -> torch.Tensor:
        temporal = self.temporal_mixer(representation)
        context = self.central(temporal, perturbation)
        personalized = temporal - context
        refined = temporal + self.refine(torch.cat((personalized, context), -1))
        spatial = self.spatial_mixer(representation - refined)
        return (self.temporal_head(temporal) + self.spatial_head(spatial)).transpose(1, 2)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"STOP expects [batch, {self.seq_len}, {self.enc_in}]")
        return self._forecast(self._representation(x_enc, x_mark_enc))

    def environment_forecasts(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None,
                              environments: int = 3) -> torch.Tensor:
        """Generate bounded GenPU branches for an external worst-loss DRO step."""
        representation = self._representation(x_enc, x_mark_enc)
        outputs = []
        for index in range(environments):
            scores = representation.square().mean(-1)
            count = max(1, self.enc_in // (index + 2))
            masked = scores.topk(count, -1).indices
            perturbation = scores.new_zeros(scores.shape).scatter(-1, masked, -1e4)
            outputs.append(self._forecast(representation, perturbation))
        return torch.stack(outputs, 1)

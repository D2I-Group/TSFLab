"""Independent TimeMosaic implementation from the published architecture.

It implements region-wise adaptive patch granularity, repeat alignment, and
segment-specific prompt-masked attention/heads. It is trained end-to-end here;
the paper's frozen foundation backbone and 321B-observation pre-training corpus
are not included.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 patch_sizes: tuple[int, ...] = (4, 8, 16), num_segments: int = 4,
                 num_heads: int = 4, dropout: float = 0.1,
                 use_revin: bool = True, selection_tau: float = 0.5, budget_weight: float = 0.001) -> None:
        super().__init__()
        sizes = tuple(sorted(set(patch_sizes)))
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_sizes, self.region_size = sizes, max(sizes)
        # A lookback that is not a multiple of the largest patch is left-padded with its first value.
        self.pad = -seq_len % self.region_size
        self.num_regions = (seq_len + self.pad) // self.region_size
        self.aligned_per_region = self.region_size // min(sizes)
        self.num_segments, self.use_revin = min(num_segments, pred_len), use_revin
        self.selection_tau, self.budget_weight = selection_tau, budget_weight
        self.aux_loss = None
        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.granularity_classifier = nn.Sequential(nn.Linear(self.region_size, d_model), nn.GELU(), nn.Linear(d_model, len(sizes)))
        self.patch_projections = nn.ModuleList(nn.Linear(size, d_model) for size in sizes)
        token_count = self.num_regions * self.aligned_per_region
        self.position = nn.Parameter(torch.empty(token_count, d_model))
        self.segment_prompts = nn.Parameter(torch.empty(self.num_segments, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        lengths = [pred_len // self.num_segments + (i < pred_len % self.num_segments) for i in range(self.num_segments)]
        self.segment_heads = nn.ModuleList(nn.Linear(token_count * d_model, length) for length in lengths)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.segment_prompts, std=0.02)

    def adaptive_patch_tokens(self, channel_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_channels = channel_history.size(0)
        regions = channel_history.reshape(batch_channels, self.num_regions, self.region_size)
        logits = self.granularity_classifier(regions)
        if self.training:
            # Hard Gumbel choice of one granularity per region (straight-through), as in the official code;
            # the budget term pulls the mean choice towards uniform use of the granularities.
            choices = torch.nn.functional.gumbel_softmax(logits, tau=self.selection_tau, hard=True, dim=-1)
            ratio = choices.reshape(-1, choices.size(-1)).mean(0)
            self.aux_loss = self.budget_weight * ((ratio - 1.0 / ratio.numel()) ** 2).mean()
        else:
            choices = torch.nn.functional.one_hot(logits.argmax(-1), len(self.patch_sizes)).to(logits.dtype)
            self.aux_loss = None
        candidates = []
        for size, projection in zip(self.patch_sizes, self.patch_projections):
            embedded = projection(regions.unfold(-1, size, size))
            repeat = self.aligned_per_region // embedded.size(2)
            candidates.append(embedded.repeat_interleave(repeat, dim=2))
        stacked = torch.stack(candidates, dim=2)
        selected = (stacked * choices[:, :, :, None, None]).sum(2)
        return selected.reshape(batch_channels, -1, selected.size(-1)) + self.position, choices

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        normalized = self.revin(x_enc, "norm") if self.use_revin else x_enc
        batch = x_enc.size(0)
        history = normalized.transpose(1, 2).reshape(batch * self.enc_in, self.seq_len)
        if self.pad:
            history = torch.cat((history[:, :1].expand(-1, self.pad), history), dim=1)
        tokens, _ = self.adaptive_patch_tokens(history)
        segments = []
        for prompt, head in zip(self.segment_prompts, self.segment_heads):
            prompt_token = prompt.reshape(1, 1, -1).expand(tokens.size(0), -1, -1)
            attended, _ = self.attention(tokens, torch.cat((prompt_token, tokens), 1), torch.cat((prompt_token, tokens), 1), need_weights=False)
            hidden = self.norm(tokens + attended)
            segments.append(head(hidden.flatten(1)))
        forecast = torch.cat(segments, dim=-1).reshape(batch, self.enc_in, self.pred_len).transpose(1, 2)
        return self.revin(forecast, "denorm") if self.use_revin else forecast

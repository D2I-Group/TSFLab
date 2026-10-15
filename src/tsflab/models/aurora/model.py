"""Compact clean-room Aurora with modality guidance and prototype flow.

This implementation is derived from the public paper equations. It accepts
optional *dense* text/image embeddings instead of bundling BERT or ViT and
returns the deterministic mean path of prototype-guided flow matching.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        patch_len: int = 16,
        num_heads: int = 4,
        num_distill_tokens: int = 2,
        num_prototypes: int = 8,
        flow_steps: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len = min(patch_len, seq_len)
        self.num_patches = math.ceil(seq_len / self.patch_len)
        self.flow_steps = flow_steps
        self.use_revin = use_revin

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.patch_position = nn.Parameter(torch.empty(self.num_patches, d_model))
        self.spectral_projection = nn.Linear(seq_len // 2 + 1, d_model)
        self.domain_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.text_queries = nn.Parameter(torch.empty(num_distill_tokens, d_model))
        self.image_queries = nn.Parameter(torch.empty(num_distill_tokens, d_model))

        self.text_distiller = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.image_distiller = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.text_guider = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.image_guider = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.guide_gate = nn.Linear(3 * d_model, 3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.future_queries = nn.Parameter(torch.empty(pred_len, d_model))
        self.condition_decoder = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.condition_norm = nn.LayerNorm(d_model)
        self.prototype_bank = nn.Parameter(torch.empty(num_prototypes, d_model))
        self.prototype_retriever = nn.Linear(3 * d_model, num_prototypes)
        self.flow_network = nn.Sequential(
            nn.Linear(2 * d_model + 1, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.output_projection = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.patch_position, std=0.02)
        nn.init.normal_(self.domain_token, std=0.02)
        nn.init.normal_(self.text_queries, std=0.02)
        nn.init.normal_(self.image_queries, std=0.02)
        nn.init.normal_(self.future_queries, std=0.02)
        # Period/trend-like deterministic bases initialize the prototype bank.
        position = torch.linspace(0, 1, self.prototype_bank.shape[1])
        bases = []
        for index in range(self.prototype_bank.shape[0]):
            frequency = index // 2 + 1
            basis = (
                torch.sin(2 * torch.pi * frequency * position)
                if index % 2 == 0
                else torch.cos(2 * torch.pi * frequency * position)
            )
            bases.append(basis)
        with torch.no_grad():
            self.prototype_bank.copy_(torch.stack(bases))

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        channel_history = x.transpose(1, 2)
        padding = self.num_patches * self.patch_len - self.seq_len
        if padding:
            channel_history = F.pad(channel_history, (padding, 0), mode="replicate")
        patches = channel_history.unfold(-1, self.patch_len, self.patch_len)
        return self.patch_embedding(patches) + self.patch_position

    def _dense_context(self, context: torch.Tensor | None, batch: int, channels: int) -> torch.Tensor:
        if context is None:
            return self.domain_token.expand(batch * channels, -1, -1)
        if context.shape[-1] != self.d_model:
            raise ValueError("dense modality embeddings must end in d_model")
        if context.ndim == 2:
            context = context.unsqueeze(1)
        if context.ndim != 3 or context.shape[0] != batch:
            raise ValueError("modality embeddings must have shape [batch, tokens, d_model]")
        return context[:, None].expand(-1, channels, -1, -1).reshape(batch * channels, context.shape[1], self.d_model)

    def encode(
        self,
        x: torch.Tensor,
        text_context: torch.Tensor | None = None,
        image_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Paper equations (1)--(21), using compact dense modality encoders."""
        check_window(x, self.seq_len, self.enc_in)
        batch, _, channels = x.shape
        normalized = self.revin(x, "norm") if self.use_revin else x
        temporal = self._patch_tokens(normalized).reshape(batch * channels, self.num_patches, self.d_model)

        text_tokens = torch.cat(
            (
                self._dense_context(text_context, batch, channels),
                temporal.mean(1, keepdim=True),
            ),
            dim=1,
        )
        spectrum = torch.fft.rfft(normalized.transpose(1, 2), dim=-1).abs()
        spectral_token = self.spectral_projection(spectrum).reshape(batch * channels, 1, self.d_model)
        if image_context is not None:
            image_tokens = torch.cat(
                (spectral_token, self._dense_context(image_context, batch, channels)),
                dim=1,
            )
        else:
            image_tokens = torch.cat(
                (
                    spectral_token,
                    self.domain_token.expand(batch * channels, -1, -1),
                ),
                dim=1,
            )

        text_query = self.text_queries.unsqueeze(0).expand(batch * channels, -1, -1)
        image_query = self.image_queries.unsqueeze(0).expand(batch * channels, -1, -1)
        distilled_text, _ = self.text_distiller(text_query, text_tokens, text_tokens)
        distilled_image, _ = self.image_distiller(image_query, image_tokens, image_tokens)
        guided_text, _ = self.text_guider(temporal, distilled_text, distilled_text)
        guided_image, _ = self.image_guider(temporal, distilled_image, distilled_image)
        pooled = torch.cat((temporal.mean(1), guided_text.mean(1), guided_image.mean(1)), dim=-1)
        weights = self.guide_gate(pooled).softmax(-1)
        fused = (
            temporal * weights[:, 0, None, None]
            + guided_text * weights[:, 1, None, None]
            + guided_image * weights[:, 2, None, None]
        )
        return self.temporal_encoder(fused), distilled_text, distilled_image

    def forecast_with_context(
        self,
        x_enc: torch.Tensor,
        *,
        text_context: torch.Tensor | None = None,
        image_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded, distilled_text, distilled_image = self.encode(
            x_enc, text_context=text_context, image_context=image_context
        )
        batch_channels = encoded.shape[0]
        queries = self.future_queries.unsqueeze(0).expand(batch_channels, -1, -1)
        condition, _ = self.condition_decoder(queries, encoded, encoded)
        condition = self.condition_norm(condition + encoded[:, -1:].expand_as(condition))

        multimodal_summary = torch.cat(
            (
                condition,
                distilled_text.mean(1, keepdim=True).expand_as(condition),
                distilled_image.mean(1, keepdim=True).expand_as(condition),
            ),
            dim=-1,
        )
        prototype_weights = self.prototype_retriever(multimodal_summary).softmax(-1)
        state = prototype_weights @ self.prototype_bank
        for step in range(self.flow_steps):
            time = state.new_full((batch_channels, self.pred_len, 1), step / self.flow_steps)
            velocity = self.flow_network(torch.cat((state, condition, time), dim=-1))
            state = state + velocity / self.flow_steps
        forecast = self.output_projection(state).squeeze(-1)
        forecast = forecast.reshape(x_enc.shape[0], self.enc_in, self.pred_len).transpose(1, 2)
        return self.revin(forecast, "denorm") if self.use_revin else forecast

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        return self.forecast_with_context(x_enc)

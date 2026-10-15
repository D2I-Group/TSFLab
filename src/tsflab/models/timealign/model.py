"""Clean-room TimeAlign from its prediction--reconstruction formulation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN


class DistributionAlignment(nn.Module):
    """Complementary local token and global relation alignment."""

    def __init__(self, local_margin: float, global_margin: float, local: bool, global_: bool) -> None:
        super().__init__()
        self.local_margin, self.global_margin = local_margin, global_margin
        self.use_local, self.use_global = local, global_

    def terms(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
        losses: dict[str, torch.Tensor] = {}
        if self.use_local:
            similarity = (prediction * target).sum(-1)
            losses["local"] = F.relu(1.0 - similarity - self.local_margin).mean()
        if self.use_global:
            pred_rel = prediction @ prediction.transpose(-1, -2)
            target_rel = target @ target.transpose(-1, -2)
            losses["global"] = F.relu((pred_rel - target_rel).abs() - self.global_margin).mean()
        return losses

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        terms = self.terms(prediction, target)
        return torch.stack(tuple(terms.values())).mean()


class PatchMLPBranch(nn.Module):
    """Channel-independent non-overlapping patch encoder and decoder."""

    def __init__(
        self,
        length: int,
        output_length: int,
        patch_num: int,
        d_model: int,
        d_ff: int,
        layers: int,
        dropout: float,
        position: bool,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.length, self.patch_num = length, patch_num
        self.patch_len = length // patch_num
        self.embedding = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, 1, patch_num, d_model)) if position else None
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d_model) if layer_norm else nn.Identity(),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(layers)
        )
        self.decoder = nn.Linear(patch_num * d_model, output_length)

    def encode(self, values: torch.Tensor) -> list[torch.Tensor]:
        patches = values.transpose(1, 2).unfold(-1, self.patch_len, self.patch_len)
        hidden = self.embedding(patches)
        if self.position is not None:
            hidden = hidden + self.position
        states = []
        for layer in self.layers:
            hidden = hidden + layer(hidden)
            states.append(hidden)
        return states

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state.flatten(-2)).transpose(1, 2)


class Model(nn.Module):
    """Dual-branch training, prediction-only inference TimeAlign."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_num: int = 4,
        d_model: int = 32,
        d_ff: int = 32,
        e_layers: int = 2,
        dropout: float = 0.1,
        pos: bool = True,
        layer_norm: bool = True,
        loc: bool = True,
        glo: bool = True,
        local_margin: float = 0.0,
        global_margin: float = 0.0,
        w_recon: float = 1.0,
        w_align: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len % patch_num or pred_len % patch_num:
            raise ValueError("patch_num must divide seq_len and pred_len")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.w_recon, self.w_align = float(w_recon), float(w_align)
        self.history_norm, self.future_norm = RevIN(enc_in, affine=False), RevIN(enc_in, affine=False)
        self.predictor = PatchMLPBranch(seq_len, pred_len, patch_num, d_model, d_ff, e_layers, dropout, pos, layer_norm)
        self.reconstructor = PatchMLPBranch(
            pred_len, pred_len, patch_num, d_model, d_ff, e_layers, dropout, pos, layer_norm
        )
        self.align_projections = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(e_layers))
        self.alignment = DistributionAlignment(local_margin, global_margin, loc, glo)

    def _forecast_states(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f"TimeAlign expects [B, {self.seq_len}, C]")
        history = self.history_norm(x_enc, "norm")
        history_states = self.predictor.encode(history)
        prediction = self.history_norm(self.predictor.decode(history_states[-1]), "denorm")
        return prediction, history_states

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        prediction, _ = self._forecast_states(x_enc)
        return prediction

    def training_objective(
        self, x_enc: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return the forecast and prediction/reconstruction/alignment loss."""
        target = target.to(device=x_enc.device, dtype=x_enc.dtype)
        prediction, history_states = self._forecast_states(x_enc)
        future = self.future_norm(target, "norm")
        future_states = self.reconstructor.encode(future)
        reconstruction = self.future_norm(self.reconstructor.decode(future_states[-1]), "denorm")
        alignment = torch.stack(
            [
                self.alignment(project(history_state), future_state.detach())
                for project, history_state, future_state in zip(self.align_projections, history_states, future_states)
            ]
        ).mean()
        prediction_loss = F.mse_loss(prediction, target)
        reconstruction_loss = F.mse_loss(reconstruction, target)
        loss = prediction_loss + self.w_recon * reconstruction_loss + self.w_align * alignment
        return (
            prediction,
            loss,
            {
                "prediction": prediction_loss,
                "reconstruction": reconstruction_loss,
                "alignment": alignment,
            },
        )

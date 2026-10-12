"""Independent LatentTSF implementation from the latent-state paradigm."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.dlinear import DLinearBackbone


class LatentStateAutoencoder(nn.Module):
    """Per-timestep observation-to-state expansion and reconstruction."""
    def __init__(self, channels: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(channels, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, channels))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        return self.decoder(states)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(values))


def latent_alignment_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus mean cosine similarity between corresponding latent states."""
    return 1.0 - F.cosine_similarity(prediction.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()


class Model(nn.Module):
    """Two-stage autoencoder pretraining and frozen latent forecasting."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 d_ff: int = 128, mse_weight: float = 10.0, cosine_weight: float = 15.0,
                 use_latent_norm: bool = True, kernel_size: int = 25, individual: bool = False,
                 ae_train_epochs: int = 100, ae_lr: float = 5e-4, ae_loss: str = "MAE") -> None:
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.mse_weight, self.cosine_weight = float(mse_weight), float(cosine_weight)
        self.ae_train_epochs, self.ae_lr, self.ae_loss = int(ae_train_epochs), float(ae_lr), ae_loss.upper()
        self.autoencoder = LatentStateAutoencoder(enc_in, d_model, d_ff)
        self.backbone = DLinearBackbone(c_in=d_model, seq_len=seq_len, pred_len=pred_len, kernel_size=kernel_size, individual=individual)
        self.latent_norm = nn.LayerNorm(d_model, elementwise_affine=False) if use_latent_norm else nn.Identity()
        self._autoencoder_frozen = False

    def pretrain(self, train_loader, device, *, stage_runner=None) -> None:
        """Stage 1: learn observation reconstruction, then freeze the map."""
        if self._autoencoder_frozen:
            return
        self.autoencoder.to(device).train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)
        criterion = nn.L1Loss() if self.ae_loss == "MAE" else nn.MSELoss()
        if stage_runner is None:
            for _ in range(self.ae_train_epochs):
                for batch in train_loader:
                    values = batch[0].float().to(device)
                    optimizer.zero_grad()
                    criterion(self.autoencoder(values), values).backward()
                    optimizer.step()
        else:
            stage_runner(self.autoencoder, train_loader, device, optimizer,
                         criterion, self.ae_train_epochs)
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)
        self.autoencoder.eval()
        self._autoencoder_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._autoencoder_frozen:
            self.autoencoder.eval()
        return self

    def _forecast_states(
        self, x_enc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f"LatentTSF expects [B, {self.seq_len}, C]")
        history_state = self.latent_norm(self.autoencoder.encode(x_enc))
        predicted_state = self.backbone(history_state)[:, -self.pred_len:]
        prediction = self.autoencoder.decode(predicted_state)
        return prediction, predicted_state

    def forward(
        self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None
    ) -> torch.Tensor:
        prediction, _ = self._forecast_states(x_enc)
        return prediction

    def training_objective(
        self, x_enc: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return the forecast and paper Equation 5 latent-space objective."""
        target = target.to(device=x_enc.device, dtype=x_enc.dtype)
        prediction, predicted_state = self._forecast_states(x_enc)
        with torch.no_grad():
            target_state = self.autoencoder.encode(target)
        prediction_mse = F.mse_loss(prediction, target)
        mse = F.mse_loss(predicted_state, target_state)
        cosine = latent_alignment_loss(predicted_state, target_state)
        loss = prediction_mse + self.mse_weight * mse + self.cosine_weight * cosine
        return prediction, loss, {
            "prediction_mse": prediction_mse,
            "latent_mse": mse,
            "latent_cosine": cosine,
        }

"""Local ST-LoRA implementation (paper Eqs. 2-4) around an LSTM backbone.

Independent rewrite from the paper; the pinned official repository was read
only to resolve omissions (initialisation, LoRA scaling, default sizes).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.marks import to_spatiotemporal


def _fit_features(values: torch.Tensor, width: int) -> torch.Tensor:
    """Slice or zero-pad the trailing feature axis to exactly ``width``."""
    if values.shape[-1] >= width:
        return values[..., :width]
    padding = values.new_zeros((*values.shape[:-1], width - values.shape[-1]))
    return torch.cat((values, padding), dim=-1)


class NodeAdaptiveLowRankLayer(nn.Module):
    """Node-Adaptive Low-rank Layer (NALL), Eq. (2) without the activation.

    For node ``i``: ``W x + b + (alpha / r) * B A_i x`` where ``W`` is frozen
    (``b`` stays trainable), ``B`` (``d_out x r``) is shared and ``A_i`` (``r x d_in``) is
    node-specific. ``A`` uses Kaiming-uniform and ``B`` zero initialisation, so
    the adaptation starts at zero; dropout acts on the low-rank path input.
    Inputs are ``[..., nodes, d_in]``.
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(num_nodes, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        for node in range(num_nodes):
            nn.init.kaiming_uniform_(self.lora_A.data[node], a=math.sqrt(5.0))
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)

    def delta_weight(self) -> torch.Tensor:
        """Per-node update ``Delta W_i = (alpha / r) B A_i``: ``[nodes, d_out, d_in]``."""
        return torch.einsum("or,nri->noi", self.lora_B, self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank = torch.einsum("...ni,nri->...nr", self.dropout(x), self.lora_A)
        return self.base(x) + F.linear(low_rank, self.lora_B) * self.scaling


class NodeSpecificPredictor(nn.Module):
    """Node-Specific Predictor (NSP), Eq. (3).

    ``H0 = Conv2D(X)`` is a 1x1 convolution over the feature axis at every
    (step, node); ``H_l = H_{l-1} + NALL_l(sigma(H_{l-1}))`` with sigma =
    RMSNorm, LeakyReLU and dropout; ``G_t`` is a NALL to one output feature.
    Input and output are ``[batch, steps, nodes, features]``.
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden: int,
        num_layers: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.start = nn.Conv2d(in_features, hidden, kernel_size=(1, 1))
        self.layers = nn.ModuleList(
            NodeAdaptiveLowRankLayer(num_nodes, hidden, hidden, rank, alpha, dropout)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.RMSNorm(hidden) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.head = NodeAdaptiveLowRankLayer(num_nodes, hidden, 1, rank, alpha, dropout)

    def activation(self, h: torch.Tensor, index: int) -> torch.Tensor:
        return self.dropout(F.leaky_relu(self.norms[index](h), negative_slope=0.1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``H0`` followed by the ``L`` residual NALL layers (before ``G_t``)."""
        h = self.start(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for index, layer in enumerate(self.layers):
            h = h + layer(self.activation(h, index))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class LSTMBackbone(nn.Module):
    """Per-node LSTM backbone (one of the paper's backbones, LargeST layout).

    Pointwise input projection, a multi-layer LSTM over each node's history,
    and a ReLU MLP from the last hidden state to all horizon steps.
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        init_dim: int,
        hid_dim: int,
        end_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, init_dim)
        self.recurrent = nn.LSTM(
            init_dim,
            hid_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.hidden = nn.Linear(hid_dim, end_dim)
        self.output = nn.Linear(end_dim, pred_len)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """``[batch, seq_len, nodes, input_dim]`` -> ``[batch, pred_len, nodes, 1]``."""
        batch, steps, nodes, width = history.shape
        sequences = history.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        encoded, _ = self.recurrent(self.input_projection(sequences))
        forecast = self.output(F.relu(self.hidden(encoded[:, -1])))
        return forecast.view(batch, nodes, -1).transpose(1, 2).unsqueeze(-1)


class Model(nn.Module):
    """ST-LoRA: LSTM backbone forecast blended with stacked node-specific predictors."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int = 2,
        num_blocks: int = 1,
        num_layers: int = 4,
        rank: int = 16,
        lora_alpha: float = 16.0,
        nsp_hidden: int = 64,
        nsp_dropout: float = 0.3,
        init_dim: int = 32,
        hid_dim: int = 64,
        end_dim: int = 512,
        backbone_layers: int = 2,
        backbone_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths, nodes, blocks, layers, rank and widths must be positive")
        if seq_len != pred_len:
            raise ValueError(
                "ST-LoRA concatenates history and adapter outputs step by step "
                "(Eq. 4), so seq_len must equal pred_len"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.input_dim = 1 + cov_dim
        self.backbone = LSTMBackbone(
            self.input_dim, pred_len, init_dim, hid_dim, end_dim, backbone_layers, backbone_dropout
        )
        self.blocks = nn.ModuleList(
            NodeSpecificPredictor(
                enc_in,
                1 if block == 0 else self.input_dim + 1,
                nsp_hidden,
                num_layers,
                rank,
                lora_alpha,
                nsp_dropout,
            )
            for block in range(num_blocks)
        )
        self.fusion = nn.Linear(self.input_dim + 1, 1)

    def adapt(self, history: torch.Tensor, base: torch.Tensor) -> list[torch.Tensor]:
        """Eq. (4) block chain: ``Z1 = H1(relu(Y_base))``, ``Zk = Hk(relu([X, Z_{k-1}]))``."""
        outputs = [self.blocks[0](F.relu(base))]
        for block in self.blocks[1:]:
            outputs.append(block(F.relu(torch.cat((history, outputs[-1]), dim=-1))))
        return outputs

    def fuse(
        self, history: torch.Tensor, base: torch.Tensor, adapted: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. (4) gate: ``R = sigmoid(F([X, mean Z]))``, ``Y = R Y_base + (1 - R) mean Z``."""
        mean = torch.stack(adapted, dim=0).mean(dim=0)
        gate = torch.sigmoid(self.fusion(torch.cat((history, mean), dim=-1)))
        return gate * base + (1.0 - gate) * mean, gate

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]")
        history = _fit_features(to_spatiotemporal(x_enc, x_mark_enc), self.input_dim)
        base = self.backbone(history)
        output, _ = self.fuse(history, base, self.adapt(history, base))
        return output.squeeze(-1)


__all__ = [
    "LSTMBackbone",
    "Model",
    "NodeAdaptiveLowRankLayer",
    "NodeSpecificPredictor",
]

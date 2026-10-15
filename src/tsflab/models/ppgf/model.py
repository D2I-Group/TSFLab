"""PPGF: probability pattern-guided forecasting (Sun et al., IEEE TNNLS 2025).

Every forecast value is first classified into one of ``K`` value groups whose
edges are equal-frequency order statistics of the training series (Eqs. 1-2),
then placed inside the predicted group by a relative offset (Eqs. 8 and 10).
The classifier works on features gated by a learned true-class-probability
confidence (TCP, Eqs. 4-6). Each channel is forecast independently with shared
weights, as the official code forecasts one univariate target.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import PositionalEmbedding


def group_edges(values: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Eqs. (1)-(2): ``[K + 1]`` edges ``A~(floor((t - 1) k / K))`` of sorted ``values``.

    Group ``k`` spans ``[edge_k, edge_{k+1})``; the first edge is the minimum and
    the last the maximum, so neighbouring groups share an edge.
    """
    ordered = values.flatten().sort().values
    count = ordered.numel()
    index = [((count - 1) * k) // num_groups for k in range(num_groups + 1)]
    return ordered[torch.tensor(index, device=ordered.device)]


def assign_groups(target: torch.Tensor, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group label and relative offset of each target value.

    ``target`` is ``[..., C]`` and ``edges`` ``[C, K + 1]``. The label counts the
    interior edges at or below the value (the first and last group are open
    towards the outside), and the offset is ``(y - left) / (right - left)``, or
    ``y - left`` for a degenerate group, as in the label of Sec. III-E.
    """
    channels = edges.shape[0]
    flat = target.reshape(-1, channels).transpose(0, 1).contiguous()  # [C, N]
    interior = edges[:, 1:-1].contiguous()
    labels = torch.searchsorted(interior, flat, right=True)  # [C, N]
    left = edges.gather(1, labels)
    width = edges.gather(1, labels + 1) - left
    shifted = flat - left
    relative = torch.where(width > 0, shifted / width.clamp_min(1e-12), shifted)
    shape = target.shape
    return (
        labels.transpose(0, 1).reshape(shape),
        relative.transpose(0, 1).reshape(shape),
    )


def place_in_group(delta: torch.Tensor, labels: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Eq. (10): ``y = delta_k (right_k - left_k) + left_k`` for the chosen group ``k``.

    ``delta`` is ``[..., C]`` (offset of the chosen group), ``labels`` the group
    index with the same shape, ``edges`` ``[C, K + 1]``. A degenerate group
    (``right_k = left_k``) adds the offset to its edge, as the official code does.
    """
    channels = edges.shape[0]
    flat_labels = labels.reshape(-1, channels).transpose(0, 1)
    flat_delta = delta.reshape(-1, channels).transpose(0, 1)
    left = edges.gather(1, flat_labels)
    width = edges.gather(1, flat_labels + 1) - left
    value = torch.where(width > 0, left + width * flat_delta, left + flat_delta)
    return value.transpose(0, 1).reshape(delta.shape)


class GatedResidualNetwork(nn.Module):
    """Gated residual network of the official code: two linear maps, GLU gate, add and norm.

    The hidden activation between the two linear maps is absent at the pinned
    revision. A residual of a different width is linearly interpolated along the
    feature axis, scaled by a learned ``2 sigmoid(m)`` gate and layer-normalized.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.d_out = d_out
        self.resample = d_in != d_out
        if self.resample:
            self.residual_gate = nn.Parameter(torch.zeros(d_out))
            self.residual_norm = nn.LayerNorm(d_out)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.glu = nn.Linear(d_hidden, 2 * d_out)
        self.norm = nn.LayerNorm(d_out)
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.glu.weight)
        nn.init.zeros_(self.glu.bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        if not self.resample:
            return x
        lead = x.shape[:-1]
        flat = x.reshape(-1, 1, x.shape[-1])
        resized = F.interpolate(flat, self.d_out, mode="linear", align_corners=True)
        resized = resized.reshape(*lead, self.d_out) * torch.sigmoid(self.residual_gate) * 2.0
        return self.residual_norm(resized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.fc1(x))
        gated = F.glu(self.glu(self.dropout(hidden)), dim=-1)
        return self.norm(gated + self.residual(x))


class VariableSelection(nn.Module):
    """Softmax-weighted sum of per-position GRNs; weights come from a GRN on all positions."""

    def __init__(self, width: int, positions: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.width = width
        self.positions = positions
        self.selector = GatedResidualNetwork(positions * width, hidden, positions, dropout)
        self.position_grns = nn.ModuleList(
            GatedResidualNetwork(width, hidden, hidden, dropout) for _ in range(positions)
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens [N, positions, width] -> (g [N, hidden], weights [N, positions])``."""
        weights = torch.softmax(self.selector(tokens.flatten(1)), dim=-1)
        encoded = torch.stack([grn(tokens[:, index]) for index, grn in enumerate(self.position_grns)], dim=1)
        return (encoded * weights.unsqueeze(-1)).sum(1), weights


class EncoderLayer(nn.Module):
    """Post-norm self-attention and feed-forward with bias-free projections.

    The official layer normalizations carry no trained affine parameters, and the
    per-head key width ``d_k`` is independent of ``d_model``.
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_ff: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.query = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.key = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.value = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.out = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape

        def heads(layer: nn.Linear) -> torch.Tensor:
            return layer(x).view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

        scores = heads(self.query) @ heads(self.key).transpose(-1, -2) / math.sqrt(self.d_k)
        context = torch.softmax(scores, dim=-1) @ heads(self.value)
        return self.out(context.transpose(1, 2).reshape(batch, length, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x + self.attention(x), x.shape[-1:])
        return F.layer_norm(x + self.ff2(F.relu(self.ff1(x))), x.shape[-1:])


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_groups: int = 4,
        d_model: int = 32,
        conv_kernel: int = 2,
        n_heads: int = 8,
        d_k: int = 64,
        d_ff: int = 2048,
        e_layers: int = 1,
        grn_hidden: int = 416,
        hidden_dim: int = 300,
        dropout: float = 0.2,
        pos_dropout: float = 0.3,
        lambda_conf: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_reg: float = 5.0,
    ) -> None:
        super().__init__()
        if seq_len <= conv_kernel:
            raise ValueError("seq_len must exceed conv_kernel")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_groups = num_groups
        self.lambda_conf = lambda_conf
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        positions = seq_len - conv_kernel + 1

        # Temporal information extractor (Sec. III-B): Conv1D, Transformer encoder, GRN.
        self.conv = nn.Conv1d(1, d_model, conv_kernel)
        self.position = PositionalEmbedding(d_model, max_len=positions)
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_k, d_ff) for _ in range(e_layers))
        self.selection = VariableSelection(d_model, positions, grn_hidden, dropout)

        # Probability pattern classifier (Sec. III-C) and relative prediction (Sec. III-D),
        # one block of K outputs per horizon step.
        self.feature = nn.Linear(grn_hidden, hidden_dim)
        self.tcp_classifier = nn.Linear(hidden_dim, pred_len * num_groups)
        self.confidence = nn.Linear(hidden_dim, pred_len)
        self.classifier = nn.Linear(hidden_dim, pred_len * num_groups)
        self.relative = nn.Linear(grn_hidden, pred_len * num_groups)

        # Group edges per channel, fitted on the training split by ``fit_groups``;
        # until then standard-normal quantiles (the runner standardizes the data).
        normal = torch.distributions.Normal(0.0, 1.0)
        levels = torch.arange(1, num_groups, dtype=torch.float32) / num_groups
        default = torch.cat([torch.tensor([-3.0]), normal.icdf(levels), torch.tensor([3.0])])
        self.register_buffer("edges", default.repeat(enc_in, 1))

    @torch.no_grad()
    def fit_groups(self, series: torch.Tensor) -> None:
        """Eqs. (1)-(2) per channel on the training series ``[T, C]``."""
        if series.ndim != 2 or series.shape[1] != self.enc_in:
            raise ValueError(f"expected a [T, {self.enc_in}] training series")
        edges = torch.stack([group_edges(series[:, c], self.num_groups) for c in range(self.enc_in)])
        self.edges.copy_(edges.to(self.edges))

    def extract(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``[B, L, C] -> g [B * C, grn_hidden]`` (one univariate series per row)."""
        series = x_enc.transpose(1, 2).reshape(-1, 1, self.seq_len)
        tokens = self.conv(series).transpose(1, 2)
        tokens = self.pos_dropout(tokens + self.position(tokens))
        for layer in self.layers:
            tokens = layer(tokens)
        return self.selection(tokens)[0]

    def heads(self, x_enc: torch.Tensor) -> dict[str, torch.Tensor]:
        """Classifier, TCP and relative outputs ``[B, H, C, K]`` (confidence ``[B, H, C]``)."""
        batch, _, channels = x_enc.shape
        horizon, groups = self.pred_len, self.num_groups
        general = self.extract(x_enc)
        feature = self.feature(general)
        confidence = self.confidence(feature)  # c_hat per horizon step
        # h_tilde = h * c_hat, then FC2 (Eq. 6); FC2 is linear, so scaling W h by c_hat is exact.
        weight = self.classifier.weight.view(horizon, groups, -1)
        bias = self.classifier.bias.view(horizon, groups)
        logits = torch.einsum("nd,hkd->nhk", feature, weight) * confidence.unsqueeze(-1) + bias
        outputs = {
            "logits": logits,
            "tcp_logits": self.tcp_classifier(feature).view(-1, horizon, groups),
            "confidence": confidence,
            "delta": F.relu(self.relative(general)).view(-1, horizon, groups),  # Eq. (8)
        }

        def arrange(value: torch.Tensor) -> torch.Tensor:
            value = value.reshape(batch, channels, horizon, *value.shape[2:])
            return value.transpose(1, 2)

        return {name: arrange(value) for name, value in outputs.items()}

    def predict(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Eq. (10) with the most probable group of the calibrated classifier."""
        labels = parts["logits"].argmax(-1)
        chosen = parts["delta"].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return place_in_group(chosen, labels, self.edges)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.predict(self.heads(x_enc))

    def pattern_loss(
        self,
        parts: dict[str, torch.Tensor],
        target: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (11): ``lambda_1 L_conf + lambda_2 L_cls + lambda_3 L_reg``.

        ``parts`` and ``target`` are aligned to the forecast channels and ``edges``
        holds their group edges. ``L_conf`` regresses ``c_hat`` on the detached
        true-class probability of the TCP classifier (Eq. 5) and trains that
        classifier with cross-entropy; ``L_cls`` is the cross-entropy of the
        calibrated classifier; ``L_reg`` sums, over groups, the squared error of
        the true group's offset averaged over the values in that group (Eq. 9).
        """
        labels, relative = assign_groups(target, edges)
        groups = self.num_groups
        tcp_logits = parts["tcp_logits"].reshape(-1, groups)
        flat_labels = labels.reshape(-1)
        true_probability = torch.softmax(tcp_logits, -1).gather(1, flat_labels[:, None]).squeeze(1)
        loss_conf = F.mse_loss(parts["confidence"].reshape(-1), true_probability.detach())
        loss_conf = loss_conf + F.cross_entropy(tcp_logits, flat_labels)
        loss_cls = F.cross_entropy(parts["logits"].reshape(-1, groups), flat_labels)
        offset = parts["delta"].gather(-1, labels.unsqueeze(-1)).squeeze(-1).reshape(-1)
        squared = (offset - relative.reshape(-1)).square()
        loss_reg = squared.new_zeros(())
        for group in range(groups):
            member = flat_labels == group
            if bool(member.any()):
                loss_reg = loss_reg + squared[member].mean()
        return self.lambda_conf * loss_conf + self.lambda_cls * loss_cls + self.lambda_reg * loss_reg

"""MambaTS: patch tokens scanned variable by variable with a learned scan order.

Paper: Cai et al., "MambaTS: Improved Selective State Space Models for Long-term
Time Series Forecasting" (arXiv 2405.16440).

Defining operations, in execution order:

* instance standardisation, then non-overlapping patches per variable embedded by a
  bias-free linear map ``z = W x`` (Sec. 5.1);
* Variable-Aware Scan along Time (VAST, Sec. 5.2): the K variables' patch tokens are
  laid out variable-major, ``[v_1 patches, v_2 patches, ...]``, so one scan visits
  every variable's whole history before the next variable;
* Temporal Mamba Blocks (TMB): selective SSM without the causal convolution and with
  dropout on the selective parameters, ``h = SSM(Dropout(Linear(z))) + sigma(Linear(z))``;
* a flatten-linear head shared across variables, inverse standardisation;
* Variable Permutation Training (VPT): the variable order is shuffled in every training
  step, and each sample's loss is credited to the K transitions its order contained,
  accumulating an asymmetric cost matrix ``P``; at inference the scan order is the
  cheapest Hamiltonian path of ``P`` (asymmetric TSP), found by simulated annealing.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def greedy_scan_path(cost: np.ndarray, start: int) -> np.ndarray:
    """Nearest-unvisited-neighbour path over a ``(K, K)`` transition cost matrix."""
    count = cost.shape[0]
    visited = np.zeros(count, dtype=bool)
    path = [start]
    visited[start] = True
    for _ in range(count - 1):
        row = np.where(visited, np.inf, cost[path[-1]])
        nxt = int(np.argmin(row))
        path.append(nxt)
        visited[nxt] = True
    return np.asarray(path, dtype=np.int64)


def anneal_scan_path(cost: np.ndarray, start: int, steps: int, seed: int) -> np.ndarray:
    """Cheapest Hamiltonian path from ``start`` by simulated annealing.

    The path cost is the sum of ``cost[a, b]`` over consecutive pairs (returning to
    the start is free). Annealing begins at the greedy path and swaps two non-start
    positions per step under a geometric cooling schedule; the best path is kept.
    """
    path = greedy_scan_path(cost, start)
    count = len(path)
    if count < 3 or steps < 1:
        return path
    rng = np.random.default_rng(seed)

    def path_cost(p: np.ndarray) -> float:
        return float(cost[p[:-1], p[1:]].sum())

    current = path_cost(path)
    best, best_cost = path.copy(), current
    spread = float(cost.std()) or 1.0
    temp0, temp_end = spread, spread * 1e-3
    decay = (temp_end / temp0) ** (1.0 / steps)
    temp = temp0
    for _ in range(steps):
        i, j = rng.integers(1, count, size=2)
        if i != j:
            path[i], path[j] = path[j], path[i]
            candidate = path_cost(path)
            delta = candidate - current
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                current = candidate
                if current < best_cost:
                    best, best_cost = path.copy(), current
            else:
                path[i], path[j] = path[j], path[i]
        temp *= decay
    return best


class TemporalMambaStack(nn.Module):
    """Pre-norm residual stack of selective SSM mixers with a final LayerNorm.

    Each layer applies ``r <- r + Mixer(LayerNorm(r))`` to the residual stream and the
    output is ``LayerNorm(r)`` (the add-then-norm ordering of the paper's blocks).
    """

    def __init__(
        self,
        d_model: int,
        layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        use_conv: bool,
    ) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
        self.mixers = nn.ModuleList(
            MambaBlock(
                d_model,
                expand * d_model,
                math.ceil(d_model / 16),
                d_conv,
                d_state,
                use_conv=use_conv,
                x_dropout=dropout,
                reference_dt_init=True,
            )
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        # Residual-branch output scaling used by the reference initialisation.
        for mixer in self.mixers:
            nn.init.kaiming_uniform_(mixer.out_proj.weight, a=math.sqrt(5))
            with torch.no_grad():
                mixer.out_proj.weight /= math.sqrt(layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        for norm, mixer in zip(self.norms, self.mixers):
            residual = residual + mixer(norm(residual))
        return self.final_norm(residual)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        e_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        patch_len: int = 48,
        stride: int = 48,
        use_causal_conv: bool = False,
        vpt_mode: int = 1,
        atsp_solver: str = "SA",
        atsp_steps: int = 20000,
        atsp_seed: int = 0,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        self.vpt_mode, self.atsp_solver = vpt_mode, atsp_solver
        self.atsp_steps, self.atsp_seed = atsp_steps, atsp_seed
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.norm = RevIN(enc_in, eps=1e-5, affine=False)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.encoder = TemporalMambaStack(d_model, e_layers, d_state, d_conv, expand, dropout, use_causal_conv)
        self.head = FlattenForecastHead(False, enc_in, d_model * self.num_patches, pred_len)

        # VAST statistics: summed per-sample loss and visit count of every ordered
        # variable pair; the diagonal holds the cost of starting a scan at a variable.
        self.register_buffer("scan_cost_sum", torch.zeros(enc_in, enc_in))
        self.register_buffer("scan_cost_count", torch.zeros(enc_in, enc_in))
        self.register_buffer("scan_updates", torch.zeros((), dtype=torch.long))
        self._transitions: torch.Tensor | None = None
        self._order_key: int | None = None
        self._order: torch.Tensor | None = None

    # ------------------------------------------------------------------ VAST
    def scan_cost_matrix(self) -> torch.Tensor:
        """Mean transition cost ``P`` shifted positive; unvisited pairs get the mean."""
        seen = self.scan_cost_count > 0
        mean = self.scan_cost_sum / self.scan_cost_count.clamp_min(1.0)
        fill = mean[seen].mean() if bool(seen.any()) else mean.new_zeros(())
        matrix = torch.where(seen, mean, fill)
        return matrix + matrix.min().abs() + 1e-7

    def update_scan_costs(self, sample_losses: torch.Tensor) -> None:
        """Credit each sample's loss to the transitions of its training-time order.

        ``sample_losses`` is ``(B,)``; it is divided by its batch standard deviation
        (the paper's non-dimensionalisation) before accumulation.
        """
        if self._transitions is None:
            raise RuntimeError("update_scan_costs requires a preceding training forward")
        losses = sample_losses.detach().float()
        if losses.numel() > 1:
            deviation = losses.std()
            if bool(deviation > 1e-12):
                losses = losses / deviation
        transitions = self._transitions
        if losses.shape[0] != transitions.shape[0]:
            raise ValueError("one loss per sample is required")
        index = transitions[..., 0] * self.enc_in + transitions[..., 1]
        flat = index.reshape(-1)
        weights = losses.unsqueeze(1).expand_as(index).reshape(-1)
        self.scan_cost_sum.view(-1).index_add_(0, flat, weights.to(self.scan_cost_sum))
        self.scan_cost_count.view(-1).index_add_(0, flat, torch.ones_like(weights).to(self.scan_cost_count))
        self.scan_updates += 1
        self._transitions = None

    @torch.no_grad()
    def scan_order(self) -> torch.Tensor:
        """Inference-time variable order: cheapest scan path of the learned costs."""
        key = int(self.scan_updates)
        if self._order is None or self._order_key != key:
            cost = self.scan_cost_matrix().double().cpu().numpy()
            start = int(np.argmin(np.diag(cost)))
            cost = cost.copy()
            np.fill_diagonal(cost, 0.0)
            if self.atsp_solver == "GD":
                path = greedy_scan_path(cost, start)
            else:
                path = anneal_scan_path(cost, start, self.atsp_steps, self.atsp_seed)
            self._order = torch.as_tensor(path, dtype=torch.long)
            self._order_key = key
        return self._order.to(self.scan_cost_sum.device)

    def _variable_order(self, batch: int, device) -> torch.Tensor | None:
        if self.vpt_mode == 0:
            return None
        if self.training:
            order = torch.argsort(torch.rand(batch, self.enc_in, device=device), dim=1)
            start = torch.cat([order[:, :1], order], dim=1)
            self._transitions = torch.stack([start[:, :-1], start[:, 1:]], dim=-1)
            return order
        return self.scan_order().unsqueeze(0).expand(batch, -1)

    # --------------------------------------------------------------- forward
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        values = self.norm(x_enc, "norm")
        patches = values.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        tokens = self.value_embedding(patches)  # (B, C, P, D)

        order = self._variable_order(batch, x_enc.device)
        if order is not None:
            gather = order[:, :, None, None].expand_as(tokens)
            tokens = torch.gather(tokens, 1, gather)
        encoded = self.encoder(tokens.flatten(1, 2)).view_as(tokens)
        if order is not None:
            restore = torch.argsort(order, dim=1)[:, :, None, None].expand_as(encoded)
            encoded = torch.gather(encoded, 1, restore)

        forecast = self.head(encoded).transpose(1, 2)  # (B, pred_len, C)
        return self.norm(forecast, "denorm")

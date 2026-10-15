"""Local ST-SSDL implementation from the paper and pinned reference details.

ST-SSDL ("How Different from the Past? Spatio-Temporal Time Series
Forecasting with Self-Supervised Deviation Learning", arXiv:2510.04908,
NeurIPS 2025) encodes the lookback window with a Chebyshev-graph-conv GRU
(paper Sec. 3.2, "spatio-temporal encoder"), queries a learnable prototype
memory with the last encoder hidden state to retrieve a soft "expected
pattern" value (Sec. 3.3, "prototype-based memory"), augments the decoder
state with that retrieval, builds an adaptive decoding graph from the
augmented representation, and autoregressively decodes ``pred_len`` steps
with a second Chebyshev-graph-conv GRU (Sec. 3.4). The paper additionally
trains two self-supervised auxiliary objectives against a distinct
*historical* reference window: a triplet contrastive loss that pulls the
query toward its nearest prototype and away from the second-nearest one, and
a deviation loss that matches the current-vs-historical query distance to
the current-vs-historical nearest-prototype distance (Sec. 3.3, Eq. for
``L_con`` / ``L_dev``).

TSFLab's runner forward contract is ``forward(x_enc, x_mark_enc, x_dec,
x_mark_dec) -> forecast``; it does not supply a second, distinct historical
reference window alongside ``x_enc``. Computing the paper's auxiliary
losses therefore needs an explicit historical window the canonical forward
signature cannot carry. Following the paper's own core forecasting path
(which only needs the *current* window's encoding to retrieve a prototype
value and decode), ``forward`` here implements the full forecasting graph
using the prototype memory's soft retrieval, and exposes the auxiliary
contrastive/deviation losses as a separate method,
:meth:`Model.auxiliary_losses`, that takes an explicit ``x_enc_hist`` /
``x_mark_enc_hist`` pair and is not called by ``forward`` or the runner. See
the "Differences" section of the model card.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.deviation_memory import PrototypeMemory, deviation_score
from tsflab.models._components.graph_conv_gru import graph_gru_step
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import coerce_time_length, future_time_features, to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


class ChebGraphConv(nn.Module):
    """Shared-weight Chebyshev graph convolution over one or more supports.

    For each support ``S`` (shape ``(N, N)`` or ``(B, N, N)``) builds the
    Chebyshev basis ``T_0=I, T_1=S, T_k=2*S*T_{k-1}-T_{k-2}``, applies every
    basis to ``x``, concatenates the results across supports and orders, and
    projects with one shared linear weight (paper Eq. for the graph
    convolution operator; local rewrite, not copied from official code).
    """

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, num_supports: int) -> None:
        super().__init__()
        self.cheb_k = cheb_k
        self.weight = nn.Parameter(torch.empty(num_supports * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out))
        nn.init.xavier_normal_(self.weight)

    def _basis(self, support: torch.Tensor) -> list[torch.Tensor]:
        """Chebyshev polynomials ``T_0 = I, T_1 = S, T_k = 2 S T_{k-1} - T_{k-2}`` up to ``cheb_k``."""
        eye = torch.eye(support.shape[-1], device=support.device, dtype=support.dtype)
        basis = [eye if support.dim() == 2 else eye.expand(support.shape[0], -1, -1), support]
        for _ in range(2, self.cheb_k):
            basis.append(2 * support @ basis[-1] - basis[-2])
        return basis[: self.cheb_k]

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        gathered = []
        for support in supports:
            # A static (N, N) support is shared by the batch; a (B, N, N) one is per sample.
            equation = "nm,bmc->bnc" if support.dim() == 2 else "bnm,bmc->bnc"
            gathered.extend(torch.einsum(equation, graph, x) for graph in self._basis(support))
        stacked = torch.cat(gathered, dim=-1)
        return torch.einsum("bni,io->bno", stacked, self.weight) + self.bias


class ChebGRUCell(nn.Module):
    """Graph-convolutional GRU cell (paper Eq. for the recurrent encoder/decoder).

    Follows the official recurrence exactly: the gate convolution produces a
    pair ``(z, r)``; ``z`` (first half) mixes into the candidate state while
    ``r`` (second half) blends the previous state with the candidate to form
    the new state. The official names are swapped relative to a textbook GRU,
    but the split positions and roles are exactly those of the shared
    ``graph_gru_step`` (first half resets, second half updates), which this
    cell calls with its two Chebyshev convolutions.
    """

    def __init__(self, dim_in: int, hidden_dim: int, cheb_k: int, num_supports: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_conv = ChebGraphConv(dim_in + hidden_dim, 2 * hidden_dim, cheb_k, num_supports)
        self.candidate_conv = ChebGraphConv(dim_in + hidden_dim, hidden_dim, cheb_k, num_supports)

    def forward(self, x: torch.Tensor, state: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        return graph_gru_step(
            x,
            state,
            lambda joined: self.gate_conv(joined, supports),
            lambda joined: self.candidate_conv(joined, supports),
        )


class Model(nn.Module):
    """ST-SSDL: prototype-memory-augmented graph-conv GRU encoder-decoder."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        rnn_units: int = 32,
        rnn_layers: int = 1,
        cheb_k: int = 2,
        prototype_num: int = 8,
        prototype_dim: int = 16,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("ST-SSDL num_nodes must be positive")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.cov_dim = input_dim - 1  # calendar covariate channels driving decoder input

        self._register_supports(adj_mx, num_nodes)

        self.encoder_cells = nn.ModuleList(
            ChebGRUCell(input_dim if layer == 0 else rnn_units, rnn_units, cheb_k, self._num_static_supports)
            for layer in range(rnn_layers)
        )

        # Reusable self-supervised prototype/deviation memory (paper-neutral component).
        self.prototype_memory = PrototypeMemory(
            query_dim=rnn_units, prototype_dim=prototype_dim, num_prototypes=prototype_num
        )

        decoder_dim = rnn_units + prototype_dim
        decoder_input_dim = output_dim + self.cov_dim
        self.decoder_cells = nn.ModuleList(
            ChebGRUCell(decoder_input_dim if layer == 0 else decoder_dim, decoder_dim, cheb_k, num_supports=1)
            for layer in range(rnn_layers)
        )
        # Data-driven decoding graph built from the prototype-augmented state (paper's hypernetwork).
        self.graph_hypernet = nn.Linear(decoder_dim, prototype_dim)
        self.output_proj = nn.Linear(decoder_dim, output_dim)

    def _register_supports(self, adj_mx: np.ndarray | None, num_nodes: int) -> None:
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, num_nodes)
        static_supports = adj_to_supports(adjacency, adj_type="symadj")
        for index, support in enumerate(static_supports):
            self.register_buffer(f"encoder_support_{index}", support)
        self._num_static_supports = len(static_supports)

    def _encoder_supports(self) -> list[torch.Tensor]:
        return [getattr(self, f"encoder_support_{index}") for index in range(self._num_static_supports)]

    def _encode(self, window: torch.Tensor) -> torch.Tensor:
        """Run the encoder over one ``(B, T, N, input_dim)`` window, returning the last hidden state."""
        supports = self._encoder_supports()
        states = [window.new_zeros(window.shape[0], self.num_nodes, cell.hidden_dim) for cell in self.encoder_cells]
        for step in window.unbind(1):
            layer_input = step
            for index, cell in enumerate(self.encoder_cells):
                states[index] = cell(layer_input, states[index], supports)
                layer_input = states[index]
        return states[-1]

    def _future_covariates(self, x_mark_dec: torch.Tensor | None, x_enc: torch.Tensor) -> torch.Tensor:
        if self.cov_dim == 0:
            return x_enc.new_zeros(x_enc.shape[0], self.pred_len, self.num_nodes, 0)
        if x_mark_dec is None:
            return x_enc.new_zeros(x_enc.shape[0], self.pred_len, self.num_nodes, self.cov_dim)
        marks = coerce_time_length(x_mark_dec, self.pred_len)
        cov = future_time_features(marks, self.num_nodes)
        if cov.shape[-1] < self.cov_dim:
            raise ValueError("ST-SSDL received fewer decoder covariate features than configured")
        return cov[..., : self.cov_dim]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec
        check_window(x_enc, self.seq_len, self.num_nodes)
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] < self.input_dim:
            raise ValueError("ST-SSDL received fewer input features than configured")
        current = history[..., : self.input_dim]

        h_t = self._encode(current)  # (B, N, rnn_units): current-window representation
        retrieval = self.prototype_memory(h_t)
        h_de = torch.cat([h_t, retrieval.value], dim=-1)  # (B, N, decoder_dim)

        node_embed = self.graph_hypernet(h_de)
        adaptive_support = torch.softmax(torch.relu(torch.einsum("bnc,bmc->bnm", node_embed, node_embed)), dim=-1)

        future_cov = self._future_covariates(x_mark_dec, x_enc)
        decoder_state = [h_de for _ in self.decoder_cells]
        go = x_enc.new_zeros(x_enc.shape[0], self.num_nodes, self.output_dim)
        outputs = []
        for step in range(self.pred_len):
            layer_input = torch.cat([go, future_cov[:, step]], dim=-1)
            for index, cell in enumerate(self.decoder_cells):
                decoder_state[index] = cell(layer_input, decoder_state[index], [adaptive_support])
                layer_input = decoder_state[index]
            go = self.output_proj(decoder_state[-1])
            outputs.append(go)
        output = torch.stack(outputs, dim=1)  # (B, pred_len, N, output_dim)
        return output.squeeze(-1)

    def auxiliary_losses(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_enc_hist: torch.Tensor,
        x_mark_enc_hist: torch.Tensor | None = None,
        margin: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Compute the paper's self-supervised contrastive/deviation losses.

        This is **not** part of the ``forward`` runner contract: it requires
        an explicit historical reference window ``x_enc_hist`` (shaped like
        ``x_enc``, e.g. the same lookback window sampled one period earlier)
        that this repository's training loop does not currently supply. A
        caller with access to such a window (e.g. a custom training script)
        may use this method directly; see the model card's "Differences"
        section.

        Returns a dict with ``contrastive_loss`` (triplet loss pulling the
        current query toward its nearest prototype and away from the
        second-nearest one) and ``deviation_loss`` (L1 distance between the
        current-vs-historical query deviation and the current-vs-historical
        nearest-prototype deviation).
        """
        if x_enc_hist.shape != x_enc.shape:
            raise ValueError("x_enc_hist must share x_enc's shape")
        current = to_spatiotemporal(x_enc, x_mark_enc)[..., : self.input_dim]
        historical = to_spatiotemporal(x_enc_hist, x_mark_enc_hist)[..., : self.input_dim]

        h_t = self._encode(current)
        h_a = self._encode(historical)
        retrieval_t = self.prototype_memory(h_t)
        retrieval_a = self.prototype_memory(h_a)

        contrastive_loss = nn.functional.triplet_margin_loss(
            retrieval_t.query.detach(), retrieval_t.nearest, retrieval_t.second_nearest, margin=margin
        )
        latent_deviation = deviation_score(retrieval_t.query, retrieval_a.query)
        prototype_deviation = deviation_score(retrieval_t.nearest, retrieval_a.nearest)
        deviation_loss = nn.functional.l1_loss(latent_deviation.detach(), prototype_deviation)
        return {"contrastive_loss": contrastive_loss, "deviation_loss": deviation_loss}


__all__ = ["Model", "ChebGraphConv", "ChebGRUCell"]

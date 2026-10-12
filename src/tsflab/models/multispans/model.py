"""MultiSPANS: multi-range spatial-temporal Transformer with structural-entropy masks (local).

Independent rewrite from Section 3 of the paper (Eqs. 3-9, Figs. 2-4) after reading the
pinned official repository (no license file). ST-tokens come from multi-size temporal
convolutions followed by multi-hop graph propagation; stacked blocks apply a temporal and
then a spatial Transformer; skip connections feed a transposed-convolution decoder.
Spatial heads are masked by the levels of an encoding tree built by greedy structural
entropy minimization (paper) and by the adjacency; the remaining heads are global.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._components.positional_encoding import positional_encoding


# --------------------------------------------------------------------------- encoding tree
def _symmetric_weights(adj: np.ndarray) -> np.ndarray:
    """Undirected, loop-free, finite weights used for structural entropy."""
    weights = np.asarray(adj, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0.0)
    return weights


class EncodingTree:
    """Rooted tree over graph vertices; vertices ``0..N-1`` are the leaves, node ``N`` the root."""

    def __init__(self, num_vertices: int) -> None:
        self.n = num_vertices
        self.parent: dict[int, int] = {v: num_vertices for v in range(num_vertices)}
        self.parent[num_vertices] = -1
        self.children: dict[int, list[int]] = {num_vertices: list(range(num_vertices))}
        self._next = num_vertices + 1

    @property
    def root(self) -> int:
        return self.n

    def is_leaf(self, node: int) -> bool:
        return node < self.n

    def vertices(self, node: int) -> list[int]:
        if self.is_leaf(node):
            return [node]
        return [v for child in self.children[node] for v in self.vertices(child)]

    def depth(self, node: int) -> int:
        d = 0
        while self.parent[node] != -1:
            node, d = self.parent[node], d + 1
        return d

    def height(self, node: int | None = None) -> int:
        node = self.root if node is None else node
        if self.is_leaf(node):
            return 0
        return 1 + max(self.height(child) for child in self.children[node])

    def combine(self, a: int, b: int) -> int:
        """Insert a new node between siblings ``a``, ``b`` and their parent."""
        parent = self.parent[a]
        node = self._next
        self._next += 1
        self.parent[node] = parent
        self.children[node] = [a, b]
        self.children[parent] = [c for c in self.children[parent] if c not in (a, b)] + [node]
        self.parent[a] = self.parent[b] = node
        return node

    def merge(self, a: int, b: int) -> int:
        """Fuse sibling ``b`` into non-leaf ``a``; a leaf ``b`` (a singleton module) moves under ``a``."""
        if self.is_leaf(a):
            raise ValueError("merge keeps a non-leaf node")
        self.children[self.parent[b]].remove(b)
        if self.is_leaf(b):
            self.parent[b] = a
            self.children[a].append(b)
            return a
        for child in self.children.pop(b):
            self.parent[child] = a
            self.children[a].append(child)
        del self.parent[b]
        return a

    def level_groups(self, level: int) -> np.ndarray:
        """Group id per vertex at ``level`` (ancestor at that depth; shallower leaves are singletons)."""
        groups = np.empty(self.n, dtype=np.int64)
        for v in range(self.n):
            path = [v]
            while self.parent[path[-1]] != -1:
                path.append(self.parent[path[-1]])
            # path[-1] is the root (depth 0); the ancestor at depth `level` is path[-1 - level].
            groups[v] = path[-1 - level] if level < len(path) - 1 else v
        return groups


def structural_entropy(adj: np.ndarray, tree: EncodingTree) -> float:
    """Eq. (3): ``-sum_{alpha != root} g_alpha / vol(G) * log2(V_alpha / V_parent)``."""
    weights = _symmetric_weights(adj)
    degree = weights.sum(axis=1)
    volume = degree.sum()
    if volume <= 0:
        return 0.0
    total = 0.0
    for node, parent in tree.parent.items():
        if parent == -1:
            continue
        inside = np.zeros(tree.n, dtype=bool)
        inside[tree.vertices(node)] = True
        v_node = degree[inside].sum()
        if v_node <= 0:
            continue
        cut = v_node - weights[inside][:, inside].sum()
        total -= cut / volume * math.log2(v_node / degree[tree.vertices(parent)].sum())
    return total


class _GreedyTreeSearch:
    """State of the greedy combine/merge search behind :func:`build_encoding_tree`."""

    def __init__(self, weights: np.ndarray, max_height: int) -> None:
        self.weights = weights
        self.n = weights.shape[0]
        self.tree = EncodingTree(self.n)
        self.degree = weights.sum(axis=1)
        self.volume = self.degree.sum()
        self.max_height = max_height
        self.member: dict[int, np.ndarray] = {}
        self.cache: dict[int, tuple[float, str | None, int, int]] = {}

    def remember(self, node: int) -> None:
        mask = np.zeros(self.n)
        mask[self.tree.vertices(node)] = 1.0
        self.member[node] = mask

    def init_statistics(self) -> None:
        for node in list(self.tree.parent):
            self.remember(node)
        self.vol = {node: float(self.degree @ m) for node, m in self.member.items()}
        self.cut = {node: self.vol[node] - float(m @ self.weights @ m) for node, m in self.member.items()}
        self.height = {node: self.tree.height(node) for node in self.tree.parent}

    def _between(self, kids: list[int]) -> np.ndarray:
        """Sibling-to-sibling weights w(a, b) via a label vector over the parent's vertices: O(n^2)."""
        label = np.full(self.n, -1)
        for column, kid in enumerate(kids):
            label[self.member[kid] > 0] = column
        inside = label >= 0
        sub_label = label[inside]
        by_row = np.zeros((len(kids), int(inside.sum())))
        np.add.at(by_row, sub_label, self.weights[np.ix_(inside, inside)])
        between = np.zeros((len(kids), len(kids)))
        np.add.at(between.T, sub_label, by_row.T)
        return between

    def _inner_cut(self, k: int) -> float:
        children = self.tree.children
        return sum(self.cut[c] for c in children[k]) if k in children else self.cut[k]

    def _kid_statistics(self, kids: list[int]):
        v = np.array([self.vol[k] for k in kids])
        g = np.array([self.cut[k] for k in kids])
        h = np.array([self.height[k] for k in kids])
        # Sum of the children's cuts; a leaf behaves as a module whose only child is itself.
        inner = np.array([self._inner_cut(k) for k in kids])
        return v, g, h, inner

    def _deltas(self, parent: int, kids: list[int], between: np.ndarray):
        volume = self.volume
        v, g, h, inner = self._kid_statistics(kids)
        v_parent = self.vol[parent]
        v_pair = v[:, None] + v[None, :]
        off_diagonal = ~np.eye(len(kids), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            combine = -2.0 * between / volume * np.log2(v_parent / v_pair)
            fits = self.tree.depth(parent) + 2 + np.maximum(h[:, None], h[None, :]) <= self.max_height
            combine = np.where(off_diagonal & fits & (v_pair > 0) & (v_pair < v_parent), combine, np.inf)
            g_new = g[:, None] + g[None, :] - 2.0 * between
            term = -g / volume * np.log2(v / v_parent)
            shift = (inner / volume)[:, None] * np.log2(v_pair / v[:, None])
            merge = -g_new / volume * np.log2(v_pair / v_parent) - term[:, None] - term[None, :] + shift + shift.T
            module = h > 0
            usable = (module[:, None] | module[None, :]) & (v[:, None] > 0) & (v[None, :] > 0)
            merge = np.where(off_diagonal & usable, merge, np.inf)
        return combine, merge

    def best_under(self, parent: int) -> tuple[float, str | None, int, int]:
        kids = self.tree.children[parent]
        if len(kids) < 2:
            return (np.inf, None, -1, -1)
        combine, merge = self._deltas(parent, kids, self._between(kids))
        best = (np.inf, None, -1, -1)
        for kind, delta in (("combine", combine), ("merge", merge)):
            flat = int(np.argmin(delta))
            if delta.flat[flat] < best[0]:
                a, b = divmod(flat, len(kids))
                best = (float(delta.flat[flat]), kind, kids[a], kids[b])
        return best

    def refresh_cache(self) -> None:
        for parent in self.tree.children:
            if parent not in self.cache:
                self.cache[parent] = self.best_under(parent)

    def _fused_cut(self, node: int) -> float:
        return float(self.member[node] @ (self.degree - self.weights @ self.member[node]))

    def _combine(self, a: int, b: int, dirty: set[int]) -> None:
        tree = self.tree
        node = tree.combine(a, b)
        self.member[node] = self.member[a] + self.member[b]
        self.vol[node] = self.vol[a] + self.vol[b]
        self.cut[node] = self._fused_cut(node)
        self.height[node] = 1 + max(self.height[a], self.height[b])
        # Subtrees below the new node moved one level down.
        stack = [a, b]
        while stack:
            x = stack.pop()
            if x in tree.children:
                dirty.add(x)
                stack.extend(tree.children[x])
        dirty.add(node)

    def _merge(self, a: int, b: int, dirty: set[int]) -> None:
        if self.tree.is_leaf(a):
            a, b = b, a
        node = self.tree.merge(a, b)
        self.member[node] = self.member[a] + self.member[b]
        self.vol[node] += self.vol[b]
        self.cut[node] = self._fused_cut(node)
        self.height[node] = 1 + max(self.height[c] for c in self.tree.children[node])
        self.cache.pop(b, None)
        dirty.add(node)

    def apply(self, parent: int, kind: str, a: int, b: int) -> None:
        dirty = {parent}
        if kind == "combine":
            self._combine(a, b, dirty)
        else:
            self._merge(a, b, dirty)
        x = parent
        while x != -1:  # ancestors: heights (and so admissible combines) may change
            self.height[x] = 1 + max(self.height[c] for c in self.tree.children[x])
            dirty.add(x)
            x = self.tree.parent[x]
        for x in dirty:
            self.cache.pop(x, None)


def build_encoding_tree(adj: np.ndarray, max_height: int, tol: float = 1e-12) -> EncodingTree:
    """Greedy structural-entropy minimization with the combine and merge operators (Sec. 3.3.3).

    Start from the flat tree (all vertices under the root). Each iteration evaluates, for
    every pair of siblings, *combine* (insert a new parent above the pair; entropy change
    ``-2 w_ab / vol * log2(V_parent / (V_a + V_b))``) and, when at least one is a non-leaf,
    *merge* (fuse the two into one node holding all their children; a leaf counts as a
    singleton module), applies the operator with the
    largest entropy reduction (first index on ties), and stops when no operator reduces the
    entropy. Operators that would make the tree taller than ``max_height`` are skipped.
    """
    search = _GreedyTreeSearch(_symmetric_weights(adj), max_height)
    if search.volume <= 0 or max_height < 2:
        return search.tree
    search.init_statistics()
    cache = search.cache
    while True:
        search.refresh_cache()
        parent = min(cache, key=lambda p: (cache[p][0], p))
        value, kind, a, b = cache[parent]
        if kind is None or value >= -tol:
            return search.tree
        search.apply(parent, kind, a, b)


def multilevel_masks(adj: np.ndarray, num_heads: int, max_height: int) -> torch.Tensor:
    """Eq. (9) head masks ``[N, N, heads]`` (True = attend).

    One mask per non-leaf tree level (pairs inside the same level-``l`` subtree, plus the
    diagonal), then the adjacency mask (positive finite weights), then unmasked heads.
    """
    tree = build_encoding_tree(adj, max_height)
    n = tree.n
    masks = []
    for level in range(1, tree.height()):
        groups = tree.level_groups(level)
        masks.append(groups[:, None] == groups[None, :])
    raw = np.asarray(adj, dtype=np.float64)
    masks.append(np.isfinite(raw) & (raw > 0))
    if len(masks) >= num_heads:
        raise ValueError("num_heads must exceed the number of masked heads (tree levels + adjacency)")
    masks.extend(np.ones((n, n), dtype=bool) for _ in range(num_heads - len(masks)))
    return torch.from_numpy(np.stack(masks, axis=-1))


def laplacian_eigenvectors(adj: np.ndarray, dim: int) -> torch.Tensor:
    """``dim`` smallest non-trivial eigenvectors of ``I - D^-1/2 A^T D^-1/2`` (zero-padded)."""
    a = np.asarray(adj, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    degree = a.sum(axis=1)
    isolated = int((degree == 0).sum())
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
    laplacian = np.eye(a.shape[0]) - (a * inv_sqrt[None, :]).T * inv_sqrt[None, :]
    values, vectors = np.linalg.eig(laplacian)
    order = np.argsort(values.real, kind="stable")
    vectors = np.real(vectors[:, order])[:, isolated + 1: isolated + 1 + dim]
    out = np.zeros((a.shape[0], dim))
    out[:, : vectors.shape[1]] = vectors
    return torch.from_numpy(out).float()


# --------------------------------------------------------------------------- network
class BatchNormChannels(nn.Module):
    """BatchNorm over the embedding axis of ``[B, N, T, d]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MultiFilterConv(nn.Module):
    """MFCL (Eqs. 4-5): replicate-padded temporal filters, then multi-hop ``D^-1 (A + I)`` propagation."""

    def __init__(self, in_dim: int, embed_dim: int, kernels: tuple[int, ...], stride: int, hops: int, alpha: float) -> None:
        super().__init__()
        width = embed_dim // (len(kernels) * (hops + 1))
        self.pads = [(round((k - 1) / 2), (k - 1) - round((k - 1) / 2)) for k in kernels]
        self.convs = nn.ModuleList(nn.Conv1d(in_dim, width, k, stride=stride) for k in kernels)
        self.hops, self.alpha = hops, alpha

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        b, n, t, c = x.shape
        flat = x.reshape(b * n, t, c).transpose(1, 2)
        parts = [conv(F.pad(flat, pad, mode="replicate")).transpose(1, 2) for pad, conv in zip(self.pads, self.convs)]
        h0 = torch.cat(parts, dim=-1).reshape(b, n, -1, parts[0].shape[-1] * len(parts))
        hops, h = [h0], h0
        for _ in range(self.hops):
            h = self.alpha * h0 + (1 - self.alpha) * torch.einsum("bntc,nm->bmtc", h, propagation)
            hops.append(h)
        return torch.tanh(torch.cat(hops, dim=-1))


class STAttention(nn.Module):
    """Multi-head attention over nodes (``kind='S'``) or time steps (``kind='T'``), Eqs. (6)-(7)."""

    def __init__(self, kind: str, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.kind, self.heads = kind, num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(embed_dim)  # official: model width, not head width

    def forward(self, value, key, query, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, t, d = query.shape
        split = (b, n, t, self.heads, d // self.heads)
        q, k, v = self.q(query).reshape(split), self.k(key).reshape(split), self.v(value).reshape(split)
        if self.kind == "S":
            score = torch.einsum("bqthd,bkthd->bqkth", q, k)
            if mask is not None:  # mask [N, N, heads] -> broadcast over batch and time
                score = score.masked_fill(~mask[None, :, :, None, :], -1e10)
            weight = self.dropout(torch.softmax(score * self.scale, dim=2))
            out = torch.einsum("bqkth,bkthd->bqthd", weight, v)
        else:
            score = torch.einsum("bnqhd,bnkhd->bnqkh", q, k)
            weight = self.dropout(torch.softmax(score * self.scale, dim=3))
            out = torch.einsum("bnqkh,bnkhd->bnqhd", weight, v)
        return self.out(out.reshape(b, n, t, d))


class STTransformer(nn.Module):
    """Post-norm Transformer layer; the attention residual adds the (position-encoded) query."""

    def __init__(self, kind: str, embed_dim: int, num_heads: int, att_dropout: float, ffn_dropout: float) -> None:
        super().__init__()
        self.attention = STAttention(kind, embed_dim, num_heads, att_dropout)
        self.drop_attention = nn.Dropout(ffn_dropout)
        self.norm_attention = BatchNormChannels(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.Dropout(ffn_dropout), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim)
        )
        self.drop_ffn = nn.Dropout(ffn_dropout)
        self.norm_ffn = BatchNormChannels(embed_dim)

    def forward(self, value, key, query, mask=None):
        x = self.norm_attention(query + self.drop_attention(self.attention(value, key, query, mask)))
        return self.norm_ffn(x + self.drop_ffn(self.ffn(x)))


class STBlock(nn.Module):
    """Temporal then spatial Transformer with outer residual BatchNorms (official forward mode 0)."""

    def __init__(self, embed_dim: int, num_heads: int, att_dropout: float, ffn_dropout: float) -> None:
        super().__init__()
        self.temporal = STTransformer("T", embed_dim, num_heads, att_dropout, ffn_dropout)
        self.spatial = STTransformer("S", embed_dim, num_heads, att_dropout, ffn_dropout)
        self.norm1 = BatchNormChannels(embed_dim)
        self.norm2 = BatchNormChannels(embed_dim)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, x, temporal_pe, spatial_pe, mask=None):
        h = self.norm1(self.temporal(x, x, x + temporal_pe) + x)
        return self.dropout(self.norm2(self.spatial(h, h, h + spatial_pe, mask) + h))


class TransposedConvDecoder(nn.Module):
    """Output layer (Sec. 3.4): transposed 1-D convolution to ``s * T_hid`` steps, tanh, then MLPs."""

    def __init__(self, embed_dim: int, hidden: int, out_dim: int, hid_len: int, out_len: int) -> None:
        super().__init__()
        self.kernel = self.stride = math.ceil(out_len / hid_len)
        k = self.kernel
        self.pad = (round((k - 1) / 2), (k - 1) - round((k - 1) / 2))
        self.trim = round((k - 1) * k / 2)
        self.tconv = nn.ConvTranspose1d(embed_dim, hidden, kernel_size=k, stride=k)
        self.time = nn.Linear(k * hid_len, out_len)
        self.channel = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, d = x.shape
        y = self.tconv(F.pad(x.reshape(b * n, t, d).transpose(1, 2), self.pad, mode="replicate"))
        y = torch.tanh(y[..., self.trim: self.trim + self.kernel * t])
        y = self.channel(self.time(y).transpose(1, 2))
        return y.reshape(b, n, -1, y.shape[-1])


def _check_config(embed_dim, conv_kernels, gconv_hops, num_heads, position_routing) -> None:
    if embed_dim % (len(conv_kernels) * (gconv_hops + 1)) or embed_dim % num_heads:
        raise ValueError("embed_dim must be divisible by len(conv_kernels) * (gconv_hops + 1) and num_heads")
    if position_routing not in ("official", "matched"):
        raise ValueError("position_routing must be 'official' or 'matched'")


def _checked_adjacency(adj_mx, num_nodes: int) -> np.ndarray:
    adjacency = np.eye(num_nodes, dtype=np.float64) if adj_mx is None else np.asarray(adj_mx, dtype=np.float64)
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("adj_mx shape must match num_nodes")
    return adjacency


class Model(nn.Module):
    """MultiSPANS forecaster: ``[B, T, N]`` history -> ``[B, H, N]`` forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        mark_features: int = 0,
        embed_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        conv_kernels: tuple[int, ...] = (1, 2, 3, 6),
        conv_stride: int = 1,
        gconv_hops: int = 3,
        gconv_alpha: float = 0.0,
        att_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        lape_ratio: float = 0.5,
        spatial_masks: bool = True,
        max_tree_height: int = 3,
        position_routing: str = "official",
    ) -> None:
        super().__init__()
        _check_config(embed_dim, conv_kernels, gconv_hops, num_heads, position_routing)
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.mark_features = mark_features
        self.position_routing = position_routing
        adjacency = _checked_adjacency(adj_mx, num_nodes)
        finite = np.where(np.isfinite(adjacency), adjacency, 0.0) + np.eye(num_nodes)
        self.register_buffer("propagation", torch.from_numpy(finite / finite.sum(axis=1, keepdims=True)).float())
        self.register_buffer(
            "head_masks",
            multilevel_masks(adjacency, num_heads, max_tree_height) if spatial_masks else torch.ones(num_nodes, num_nodes, num_heads, dtype=torch.bool),
        )
        lape_dim = max(1, int(round(num_nodes * lape_ratio)))
        self.register_buffer("laplacian", laplacian_eigenvectors(adjacency, lape_dim))
        self.node_pe = nn.Linear(lape_dim, embed_dim)

        self.encoder = MultiFilterConv(1 + mark_features, embed_dim, tuple(conv_kernels), conv_stride, gconv_hops, gconv_alpha)
        self.hid_len = math.ceil(seq_len / conv_stride)
        self.time_pe = positional_encoding("sincos", False, self.hid_len, embed_dim)
        self.blocks = nn.ModuleList(STBlock(embed_dim, num_heads, att_dropout, ffn_dropout) for _ in range(num_layers))
        self.dropout = nn.Dropout(ffn_dropout)
        self.decoder = TransposedConvDecoder(embed_dim, embed_dim // 2, 1, self.hid_len, pred_len)

    def inputs(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """``[B, T, N]`` (+ covariates) -> ``[B, N, T, 1 + mark_features]``."""
        if self.mark_features:
            data = to_spatiotemporal(x_enc, x_mark_enc)
            if data.shape[-1] != 1 + self.mark_features:
                raise ValueError(f"expected {self.mark_features} covariate features, got {data.shape[-1] - 1}")
        else:
            data = x_enc[..., None]
        return data.permute(0, 2, 1, 3)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"expected x_enc [batch, {self.seq_len}, {self.num_nodes}], got {tuple(x_enc.shape)}")
        node_pe = self.node_pe(self.laplacian)[None, :, None, :]  # D_s
        time_pe = self.time_pe[None, None, :, :]  # D_t
        if self.position_routing == "official":  # Sec. 3.3.1 / official STBlock
            temporal_pe, spatial_pe = node_pe, time_pe
        else:  # Sec. 3.3.2
            temporal_pe, spatial_pe = time_pe, node_pe
        x = self.encoder(self.inputs(x_enc, x_mark_enc), self.propagation)
        skip = x
        for block in self.blocks:
            h = block(x, temporal_pe, spatial_pe, self.head_masks)
            skip = skip + h
            x = x + h
        out = self.decoder(self.dropout(skip))  # [B, N, H, 1]
        return out[..., 0].transpose(1, 2)


__all__ = [
    "EncodingTree",
    "Model",
    "build_encoding_tree",
    "laplacian_eigenvectors",
    "multilevel_masks",
    "structural_entropy",
]

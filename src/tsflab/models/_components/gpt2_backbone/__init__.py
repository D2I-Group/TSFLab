"""Decoder-only GPT-2 Transformer used as a pretrained-LLM backbone.

LLM-based forecasters feed their own time-series token embeddings into a
pretrained GPT-2 and read its final hidden states. This module implements the
GPT-2 (Radford et al., 2019) architecture independently: learned absolute
positions added to the input embeddings, pre-LayerNorm causal self-attention
and GELU (tanh approximation) MLP blocks, and a final LayerNorm. It never
downloads anything; ``load_gpt2_weights`` reads an explicitly supplied,
checksum-verified local ``model.safetensors`` with a small built-in reader so
no extra runtime dependency is needed. The token-embedding table of the
checkpoint is not used: callers always pass ``inputs_embeds``.

Two explicit options, off by default, serve partly fine-tuned trunks: a LoRA
adapter added to the fused query/key/value projection (``lora_rank > 0``) and
the fused ``scaled_dot_product_attention`` kernel
(``attn_implementation="sdpa"``) in place of the explicit masked softmax.
``forward_with_hidden_states`` also returns the per-block hidden states.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPT2Config:
    """Architecture hyper-parameters; defaults are the released GPT-2 small."""

    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    n_positions: int = 1024
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    attn_implementation: str = "eager"
    lora_rank: int = 0
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0

    def __post_init__(self) -> None:
        self._check_sizes()
        self._check_regularization()

    def _check_sizes(self) -> None:
        if min(self.n_layer, self.n_embd, self.n_head, self.n_positions) < 1:
            raise ValueError("GPT-2 sizes must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")

    def _check_regularization(self) -> None:
        if not all(0.0 <= p < 1.0 for p in (self.embd_pdrop, self.attn_pdrop, self.resid_pdrop)):
            raise ValueError("dropout probabilities must be in [0, 1)")
        if self.attn_implementation not in ("eager", "sdpa"):
            raise ValueError("attn_implementation must be 'eager' or 'sdpa'")
        if self.lora_rank < 0:
            raise ValueError("lora_rank must be non-negative (0 disables LoRA)")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


class LoRAAdapter(nn.Module):
    """Low-rank update ``(alpha / r) * B(A(dropout(x)))`` added to a frozen projection."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / float(rank)
        # Standard LoRA start: A Kaiming-uniform, B zero, so the adapted layer
        # equals the pretrained layer at initialisation.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scale


class GPT2Attention(nn.Module):
    """Causal multi-head self-attention with a fused query/key/value projection.

    With ``config.lora_rank > 0`` a ``LoRAAdapter`` (submodule ``lora``) adds a
    low-rank update to the fused projection: ``qkv = c_attn(x) + lora(x)``.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attn_implementation = config.attn_implementation
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.lora = (
            LoRAAdapter(config.n_embd, 3 * config.n_embd, config.lora_rank, config.lora_alpha, config.lora_dropout)
            if config.lora_rank
            else None
        )
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.c_attn(hidden)
        if self.lora is not None:
            qkv = qkv + self.lora(hidden)
        query, key, value = qkv.split(width, dim=-1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        # softmax(Q K^T / sqrt(d_head) + causal mask) V
        if self.attn_implementation == "sdpa":
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
            future = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
            scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
            weights = self.attn_dropout(torch.softmax(scores, dim=-1))
            context = weights @ value
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.c_proj(context))


class GPT2MLP(nn.Module):
    """Position-wise 4x expansion MLP with the tanh-approximated GELU."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(hidden), approximate="tanh")))


class GPT2Block(nn.Module):
    """Pre-norm residual block: ``h + attn(ln_1(h))`` then ``h + mlp(ln_2(h))``."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.ln_1(hidden))
        return hidden + self.mlp(self.ln_2(hidden))


def _init_gpt2_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=0.02)
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.bias)


class GPT2Backbone(nn.Module):
    """GPT-2 trunk mapping ``inputs_embeds [B, T, n_embd]`` to final hidden states.

    ``forward`` adds learned positions ``wpe[0:T]``, applies embedding dropout,
    the causal blocks ``h`` and the final LayerNorm ``ln_f`` and returns
    ``[B, T, n_embd]``. Parameter names follow the released checkpoint layout
    (``wpe``, ``h.<i>.ln_1``, ``h.<i>.attn.c_attn``, ``ln_f``, ...), with the
    checkpoint's transposed ``Conv1D`` weights stored as ``nn.Linear``.
    """

    def __init__(self, config: GPT2Config | None = None) -> None:
        super().__init__()
        self.config = config or GPT2Config()
        cfg = self.config
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList(GPT2Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """GPT-2 initialization: N(0, 0.02) weights, residual projections / sqrt(2 L).

        LoRA adapters keep their own start (Kaiming ``A``, zero ``B``).
        """
        adapters = self._adapter_module_ids()
        for module in self.modules():
            if id(module) not in adapters:
                _init_gpt2_module(module)
        for block in self.h:
            for proj in (block.attn.c_proj, block.mlp.c_proj):
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def _adapter_module_ids(self) -> set[int]:
        return {id(m) for a in self.modules() if isinstance(a, LoRAAdapter) for m in a.modules()}

    def _embed(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.n_embd:
            raise ValueError(f"inputs_embeds must be [batch, time, {self.config.n_embd}]")
        length = inputs_embeds.shape[1]
        if length > self.config.n_positions:
            raise ValueError(f"sequence length {length} exceeds n_positions {self.config.n_positions}")
        positions = torch.arange(length, device=inputs_embeds.device)
        return self.drop(inputs_embeds + self.wpe(positions))

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden = self._embed(inputs_embeds)
        for block in self.h:
            hidden = block(hidden)
        return self.ln_f(hidden)

    def forward_with_hidden_states(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Final state and the ``n_layer + 1`` hidden states ``[B, T, n_embd]``.

        The list holds the input of every block (``h_0`` is the position-embedded,
        dropped-out input) followed by the final normalized state
        ``ln_f(h_L)``, the Hugging Face ``output_hidden_states`` convention.
        """
        hidden = self._embed(inputs_embeds)
        states = []
        for block in self.h:
            states.append(hidden)
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        states.append(hidden)
        return hidden, states


_SAFETENSORS_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
_CONV1D_WEIGHTS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def read_safetensors(path: str | Path, keep=lambda name: True) -> dict[str, torch.Tensor]:
    """Read selected tensors from a local ``.safetensors`` file (no network).

    Format: an 8-byte little-endian header length, a JSON header mapping each
    name to ``dtype``, ``shape`` and byte ``data_offsets``, then the raw data.
    """
    tensors: dict[str, torch.Tensor] = {}
    with Path(path).open("rb") as handle:
        (header_size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_size))
        base = 8 + header_size
        for name, info in header.items():
            if name == "__metadata__" or not keep(name):
                continue
            tensors[name] = _read_safetensor(handle, base, name, info)
    return tensors


def _read_safetensor(handle, base: int, name: str, info: dict) -> torch.Tensor:
    dtype = _SAFETENSORS_DTYPES.get(info["dtype"])
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype {info['dtype']!r} for {name!r}")
    start, end = info["data_offsets"]
    handle.seek(base + start)
    raw = bytearray(handle.read(end - start))
    flat = torch.frombuffer(raw, dtype=dtype) if raw else torch.empty(0, dtype=dtype)
    return flat.reshape(info["shape"]).clone()


def _gpt2_state_from_checkpoint(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {}
    for name, tensor in raw.items():
        name = name.removeprefix("transformer.")
        if name.endswith(_CONV1D_WEIGHTS):
            tensor = tensor.t().contiguous()
        state[name] = tensor
    return state


def _expected_gpt2_state(backbone: GPT2Backbone) -> tuple[set[str], dict[str, torch.Tensor]]:
    """LoRA adapter keys and the remaining (checkpoint-backed) state of ``backbone``."""
    adapter_keys = {name for name in backbone.state_dict() if ".attn.lora." in name}
    expected = {k: v for k, v in backbone.state_dict().items() if k not in adapter_keys}
    return adapter_keys, expected


def _check_gpt2_state(state: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    missing = sorted(set(expected) - set(state))
    if missing:
        raise KeyError(f"GPT-2 checkpoint is missing tensors: {', '.join(missing[:5])}")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"GPT-2 checkpoint tensor {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected[name].shape)}"
            )


def load_gpt2_weights(backbone: GPT2Backbone, path: str | Path) -> GPT2Backbone:
    """Load released GPT-2 weights into ``backbone`` in place and return it.

    Reads ``wpe``, ``ln_f`` and the first ``backbone.config.n_layer`` blocks
    (so a truncated trunk takes the checkpoint's first layers), transposing the
    checkpoint's ``Conv1D`` ``[in, out]`` weights to ``nn.Linear`` ``[out, in]``.
    Keys may carry a ``transformer.`` prefix; the causal-mask buffers and the
    token table are ignored. Missing tensors or shape mismatches raise. LoRA
    adapter tensors (``lora_rank > 0``) are not in the checkpoint and keep
    their current values.
    """
    n_layer = backbone.config.n_layer

    def wanted(name: str) -> bool:
        name = name.removeprefix("transformer.")
        if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
            return False
        if name.startswith("h."):
            return int(name.split(".", 2)[1]) < n_layer
        return name.split(".", 1)[0] in {"wpe", "ln_f"}

    state = _gpt2_state_from_checkpoint(read_safetensors(path, wanted))
    adapter_keys, expected = _expected_gpt2_state(backbone)
    _check_gpt2_state(state, expected)
    if not adapter_keys:
        backbone.load_state_dict(state, strict=True)
        return backbone
    incompatible = backbone.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != adapter_keys:
        raise KeyError(f"GPT-2 checkpoint does not match the trunk: {incompatible}")
    return backbone


__all__ = [
    "GPT2Attention",
    "GPT2Backbone",
    "GPT2Block",
    "GPT2Config",
    "GPT2MLP",
    "LoRAAdapter",
    "load_gpt2_weights",
    "read_safetensors",
]

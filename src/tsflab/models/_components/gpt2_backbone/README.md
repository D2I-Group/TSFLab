---
name: "gpt2_backbone"
description: "Decoder-only GPT-2 trunk over input embeddings (learned positions, causal pre-norm blocks), optional LoRA on fused QKV, and an offline safetensors loader. Use for LLM-based forecasters reusing pretrained GPT-2 weights; not for causal Transformers trained from scratch or released time-series foundation models."
---

# gpt2_backbone

## What it does

LLM-based forecasters replace GPT-2's token table with their own time-series
token embeddings and use the pretrained Transformer trunk as a sequence model.
`GPT2Backbone(config)` is that trunk:

`h_0 = dropout(E + wpe[0:T])`, `h_l = h + attn(ln_1(h))`, then `h + mlp(ln_2(h))`,
output `ln_f(h_L)`, where `attn` is causal multi-head attention with a fused
`c_attn` query/key/value projection and scale `1/sqrt(d_head)`, and `mlp` is
`c_proj(gelu_tanh(c_fc(x)))` with 4x expansion.

`load_gpt2_weights(backbone, path)` fills it from a released checkpoint.
Optional, off by default: a LoRA update on the fused projection,
`qkv = c_attn(x) + (alpha / r) B(A(dropout(x)))`, and the fused
`scaled_dot_product_attention` kernel for the same causal softmax.

## When to use

Use when a forecaster feeds its own embeddings to a pretrained GPT-2 trunk (frozen, partly tuned, or adapted). Do not use for a generic causal Transformer trained from scratch (prefer `transformer_encdec` or `tst_transformer`), or for released time-series foundation models (use the `_foundation` runtime boundary).

## Interface

- `GPT2Config(n_layer=12, n_embd=768, n_head=12, n_positions=1024, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, layer_norm_epsilon=1e-5, attn_implementation="eager", lora_rank=0, lora_alpha=1.0, lora_dropout=0.0)`: frozen dataclass; defaults are GPT-2 small (124M) without adapters. Non-positive sizes, `n_embd % n_head != 0`, dropout or `lora_dropout` outside `[0, 1)`, `lora_rank < 0`, or an `attn_implementation` other than `"eager"`/`"sdpa"` raise `ValueError`.
- `GPT2Backbone(config=None)`: `forward(inputs_embeds [B, T, n_embd]) -> [B, T, n_embd]`; wrong rank/width or `T > n_positions` raise `ValueError`. `forward_with_hidden_states(inputs_embeds) -> (final [B, T, n_embd], states)` where `states` holds `n_layer + 1` tensors `[B, T, n_embd]`: the input of every block (`h_0` after positions and dropout) and then `ln_f(h_L)` (the Hugging Face `output_hidden_states` convention). `reset_parameters()` is called by the constructor; subclasses may override it. Submodules `wpe`, `drop`, `h` (`ModuleList` of `GPT2Block`), `ln_f`; block children `ln_1`, `attn.c_attn`, `attn.c_proj`, `ln_2`, `mlp.c_fc`, `mlp.c_proj`. No token table (`wte`), no LM head, no KV cache, no padding mask. Random initialization: `N(0, 0.02)` weights, zero biases, residual projections scaled by `1/sqrt(2 n_layer)`.
- `GPT2Block(config)`: one pre-norm residual block, `forward([B, T, n_embd]) -> [B, T, n_embd]`.
- `GPT2Attention(config)`: causal multi-head self-attention (`c_attn` fused `n_embd -> 3 n_embd`, `c_proj`, attention-probability and residual dropout); `forward([B, T, n_embd])`. `"eager"` masks scores with the dtype minimum before softmax and applies `attn_dropout` to the weights; `"sdpa"` calls `F.scaled_dot_product_attention(is_causal=True, dropout_p=attn_pdrop in training)` (same mathematics; floating-point rounding and the dropout random stream differ from eager). With `lora_rank > 0` the submodule `lora` (`LoRAAdapter`) is registered after `c_proj` and `qkv = c_attn(x) + lora(x)`; otherwise `lora is None` and no parameters are added.
- `LoRAAdapter(in_features, out_features, rank, alpha, dropout)`: `forward(x) = lora_B(lora_A(dropout(x))) * alpha / rank`; `lora_A` Kaiming-uniform (`a = sqrt(5)`), `lora_B` zero, both bias-free, so an adapted trunk starts equal to the plain one. `GPT2Backbone.reset_parameters` leaves adapters untouched.
- `GPT2MLP(config)`: `dropout(c_proj(gelu_tanh(c_fc(x))))` with `c_fc: n_embd -> 4 n_embd`.
- `load_gpt2_weights(backbone, path) -> backbone`: reads `wpe`, `ln_f`, and blocks `0..n_layer-1` (a shorter trunk takes the checkpoint's first layers), strips an optional `transformer.` prefix, skips `wte` and the `attn.bias`/`attn.masked_bias` mask buffers, transposes `Conv1D` `[in, out]` weights to `Linear` `[out, in]`, and loads strictly (LoRA adapter tensors, absent from checkpoints, keep their values). Missing tensors raise `KeyError`; shape mismatches raise `ValueError`. Does not change `requires_grad`.
- `read_safetensors(path, keep=...) -> dict[str, Tensor]`: dependency-free reader for F32/F16/BF16 tensors, reading only names accepted by `keep`.
- Dropout (embedding, attention probabilities, residual) is active in training mode, as in the released model; call `.eval()` for deterministic output.

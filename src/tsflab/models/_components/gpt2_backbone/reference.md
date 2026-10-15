# gpt2_backbone — reference

## Origin and granularity

Extracted with AutoTimes (frozen full GPT-2) and LLM4TS (first six GPT-2 layers
with LoRA on `attn.c_attn` and LayerNorm tuning), the first catalog entries that
use a pretrained language-model backbone. The cut is the GPT-2 trunk and its weight
loader only: input embeddings, heads, freezing policy, and adapters such as
most adapters stay in the models because they are paper-specific. `Linear` layers
(not GPT-2 `Conv1D`) are used, so adapters can wrap `attn.c_attn` directly
(LLM4TS does this with its own `LoRALinear`).

CALF (Liu et al., "CALF: Aligning LLMs for Time Series Forecasting via
Cross-modal Fine-Tuning", AAAI 2025) migrated its former model-local GPT-2
here; its layout put the LoRA adapter beside `c_attn` (`attn.lora.lora_A`,
`attn.lora.lora_B`), used the fused attention kernel, and read the per-block
hidden states for its feature-alignment loss. Those became the explicit
`lora_*`, `attn_implementation`, and `forward_with_hidden_states` options so
CALF's checkpoints and numerics are unchanged; CALF keeps its own random
initialization (a model-local `reset_parameters` override). AutoTimes ("AutoTimes:
Autoregressive Time Series Forecasters via Large Language Models", NeurIPS
2024), LLM4TS ("LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient
Time-Series Forecasters", ACM TIST), and TALON (arXiv 2508.07195) use the
defaults.
Nothing is downloaded here: weights are a model's declared, checksum-pinned
`ModelArtifact`, fetched explicitly and handed to `artifact_factory`.

## Invariants and equivalence evidence

- Checks at extraction: shapes, parameter names, validation; one block checked against the explicitly written GPT-2 equations; causality (changing later inputs leaves earlier outputs unchanged); loading a safetensors file in the released layout with `transformer.` prefix and `Conv1D` orientation reproduces the source model, truncation takes the first layers, missing tensors raise; reader round trip and filtering.
- Options were checked for: default options add no adapter and keep the parameter layout; the SDPA kernel matches the eager masked softmax; the LoRA adapter starts as identity and survives reset; LoRA adds a low-rank update to the fused projection; hidden states follow the `output_hidden_states` convention; the loader leaves LoRA adapters in place; option validation. These checks passed in the full suite run of 2026-10-03 before the test suite was consolidated. The defaults leave the module tree, parameter registration, initialization stream, and the eager forward of AutoTimes, LLM4TS, and TALON unchanged (the eager branch is the former code; `lora` is `None`).
- CALF migration: equivalence against a frozen verbatim copy of the pre-migration CALF was checked at migration: built under the same seed as the migrated CALF, the two had identical state-dict keys and values, strict cross-loading, eval forecasts and hidden features, and training-mode outputs, loss, and every parameter gradient (rtol = atol = 0), with random and with loaded GPT-2 weights. That frozen-copy test passed in the full suite run of 2026-10-03 before the test suite was consolidated.
- One-off reference check (not part of CI, which has no `transformers`): a randomly initialized Hugging Face `GPT2Model` (transformers 5.18.0, 3 layers, width 64) saved with `save_pretrained` and loaded with `load_gpt2_weights` gave the same `last_hidden_state` for `inputs_embeds` to within 5e-7, for the full and a two-layer truncated trunk.
- Released checkpoint facts: `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, `model.safetensors` SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`, keys without prefix (`h.<i>.attn.c_attn.weight` `[768, 2304]`, ...), config `gelu_new`, `layer_norm_epsilon=1e-5`, dropouts 0.1.

## Variants and options

Configuration fields above. `attn_implementation`: `"eager"` (default; explicit masked softmax) or `"sdpa"` (fused kernel, used by CALF). `lora_rank` (0 = no adapter, the default), `lora_alpha`, `lora_dropout`: a LoRA adapter beside every block's `c_attn` (used by CALF; LLM4TS instead wraps `c_attn` with its local `LoRALinear`, whose state-dict layout `c_attn.base.*` differs, so it stays local). Initialization is overridable through `reset_parameters` (CALF scales residual projections by the released depth 12 and draws in parameter order). Not covered: other LLM families (LLaMA, OPT, ...), the token table and LM head, attention masks for padding, KV caching, or half-precision loading policies.

## Related components

`tst_transformer` and `transformer_encdec` (Transformer stacks trained from scratch), `embed` (input embeddings that can feed this trunk), `revin` (instance normalization typically applied before tokenization).

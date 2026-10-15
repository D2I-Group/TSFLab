# CALF — reference

## Paper

Peiyuan Liu, Hang Guo, Tao Dai, Naiqi Li, Jigang Bao, Xudong Ren, Yong Jiang, Shu-Tao Xia. "CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning." AAAI 2025 (arXiv:2403.07300; the repository was originally named LLaTA).

```bibtex
@inproceedings{liu2025calf,
  author    = {Peiyuan Liu and Hang Guo and Tao Dai and Naiqi Li and Jigang Bao and Xudong Ren and Yong Jiang and Shu-Tao Xia},
  title     = {{CALF}: Aligning {LLMs} for Time Series Forecasting via Cross-modal Fine-Tuning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {18},
  pages     = {18915--18923},
  year      = {2025},
  doi       = {10.1609/aaai.v39i18.34082}
}
```

## Implementation mapping

- Official files inspected: `models/CALF.py`, `models/GPT2_arch.py`, `pca.py`, `utils/cmLoss.py`, `exp/exp_long_term_forecasting.py`, `run.py`, `scripts/long_term_forecasting/*.sh`.
- `CALFGPT2Backbone` is the shared `gpt2_backbone` trunk (pre-LN, causal attention over the channel-token axis, tanh-GELU MLP, fused attention kernel; options `attn_implementation="sdpa"`, `lora_rank`, `forward_with_hidden_states`) with CALF's own random initialization and the component's `LoRAAdapter`. The former model-local copy was migrated without changing state-dict keys, initialization, outputs, or gradients.
- `principal_embeddings` is the buffer `D_hat = PCA(D)`; `spec.build_pretrained` loads the checksum-verified `gpt2` artifact.
- `cross_modal_loss`: `feature_weight * sum gamma^(L-l) sim(phi_time(F_time^l), phi_text(F_text^l))` over the `L + 1` hidden features plus `output_weight * sim(Y_time, Y_text)`, added through the spec's `training_objective`.

## Differences in detail

- Pretrained backbone: GPT-2 base weights are a declared `ModelArtifact` (`gpt2`, `hf://openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors`, MIT, SHA-256 pinned, optional), never downloaded implicitly or vendored. Without the file, construction stays offline and both branches start from GPT-2-style random initialization with the same freezing; such results should not be reported as CALF.
- Loss weights: the paper states `lambda_1 = 1` (feature) and `lambda_2 = 0.01` (output); the official defaults, never overridden by its scripts, are `feature_w = 0.01` and `output_w = 1.0`. The preset follows the code; `feature_decay = 0.8` is the paper's `gamma`.
- Similarity and supervised losses: the paper uses L1 for all three terms on ETT and smooth L1 on Weather/ECL/Traffic. Here the supervised term is the configured criterion and `alignment_loss` (`l1`, `smooth_l1`, `mse`) selects the similarity. Short-term (M4) SMAPE/MASE variants are not reproduced.
- Principal word embeddings: the official offline `pca.py` fits scikit-learn PCA with 500 components on the transposed word table; exact SVD computes the same scores (checked against scikit-learn up to component signs), stored as a buffer.
- Optimization: the official code steps the projection layers with a second Adam at the same learning rate, lr 5e-4, cosine or type1 schedules; TSFLab uses one configured optimizer and schedule. Frozen weights have `requires_grad = False`.
- Unspecified settings follow the official code: 12 heads, 2048-wide ReLU feed-forward, dropout 0.1 in the match module; GPT-2 dropout 0.1; the scripts' `--n_heads`, `--d_ff`, `--dropout` are unused officially and have no counterpart.
- Scope: long- and short-horizon point forecasting only.

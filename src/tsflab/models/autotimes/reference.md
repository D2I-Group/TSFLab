# AutoTimes — reference

## Citation

```bibtex
@inproceedings{liu2024autotimes,
  title     = {{AutoTimes}: Autoregressive Time Series Forecasters via Large Language Models},
  author    = {Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

## Differences in detail

Paper (arXiv 2402.02370, Sec. 3 and Appendix A) and the official code at revision `9ff9aac` (`models/AutoTimes_Gpt2.py`, `layers/mlp.py`, `exp/exp_long_term_forecasting.py`, `scripts/method_generality/gpt2.sh`; MIT) were read; nothing was copied. A one-off check against the official `AutoTimes_Gpt2.Model` (randomly initialized two-layer Hugging Face `GPT2Model`, transformers 5.18.0), after copying its parameters into the local model, matched the next-segment outputs, the official rolling-test loop, and the training MSE to within 6e-7.

- Backbone: the paper's default LLM is LLaMA-7B; this entry implements the official GPT-2 variant (the paper's method-generality experiment). The trunk is the shared local `gpt2_backbone` re-implementation; the released `openai-community/gpt2` `model.safetensors` (revision `607a30d`, SHA-256 pinned) is a required artifact fetched with `tsf model artifacts AutoTimes --fetch gpt2`. Without it, `spec.build` constructs a randomly initialized trunk for offline structure tests only; runs refuse to start.
- Timestamp position embeddings (`--mix_embeds`, Eq. 4-5) need LLM embeddings of textual timestamps precomputed with the LLM tokenizer, which TSFLab's numeric marks do not carry; they are not implemented. The official GPT-2 script also runs without them.
- The official loader samples one channel per example; here a batch holds all channels, flattened into the batch axis (same computation).
- Training uses the configured loss (the paper uses MSE) on next-segment targets; when `pred_len < token_len`, only the available future values of the last segment are supervised. With `features = "MS"` only the target channel enters the objective. The context length is the task `seq_len` at training and inference (official scripts can test with a different `test_seq_len`).
- The official `MLP` treats `mlp_hidden_layers = 1` like 2; here 1 is rejected. `llm_layers` (default 12, the full GPT-2) runs a truncated trunk for cheap structure tests.
- Default preset: `token_len = 96`, 2-layer tanh MLPs with `mlp_hidden_dim = 512` and dropout 0.1, from `scripts/method_generality/gpt2.sh` (context 672 = 7 segments). The official optimizer settings (Adam, lr 2e-3, cosine schedule, batch 2048, 10 epochs, AMP) belong to the run config.

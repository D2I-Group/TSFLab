# MambaTS — reference

## Differences in detail

Checked against the official repository at the pinned revision (`models/MambaTS.py`,
`layers/mamba_ssm/mixer2_seq_simple.py`, `layers/mamba_ssm/mamba_simple.py`,
`exp/exp_long_term_forecasting.py`, `run.py`) and the paper's Sec. 5 and Eqs. 3-7. Inputs are
`[B, seq_len, enc_in]`, output `[B, pred_len, enc_in]` (the runner slices the target for `MS`). The SSM mixer,
instance normalisation and head are shared components; the VAST cost bookkeeping and ATSP decoder are model-local.

- **Kernels and dependencies.** The official model needs `mamba_ssm` kernels, Triton fused add-norm and `python_tsp`; here the scan is the `mamba` component's sequential PyTorch recurrence (equivalent, slower) and the add-then-LayerNorm residual stream is explicit.
- **Causal convolution.** Official experiments never pass `--use_casual_conv`, so no convolution (and no SiLU on the scan input) is used; `use_causal_conv=False` is the default, as in the paper's TMB.
- **Cost accumulation.** Official code keeps a running sum and visit count (mean of observed costs, as in Proposition 2); the paper's Eq. 6 describes a moving average with rate beta whose batch form is unspecified, so it is not implemented. Unvisited pairs (NaN in the official division) take the mean of visited pairs; the cheapest pair is shifted to about zero as in the official `get_adjacency_matrix`.
- **Per-sample attribution.** The official `scatter_add_` pairs flattened `(batch, transition)` indices with `cost.repeat(K)`, which credits sample `i % B`'s loss to the wrong samples' transitions; here each sample's loss goes to its own K transitions (the paper's intent). The batch-deviation scaling is skipped for a single sample or zero spread (NaN or infinity officially). The unused official `ending_points` is dropped and the statistics are persistent buffers, not frozen parameters.
- **Loss.** The official VPT mode switches to element-wise MSE to read per-sample losses; here the configured criterion (MSE in the paper) is applied per sample without gradients and the optimised loss is that criterion on the whole batch.
- **ATSP solver.** Official: `python_tsp` simulated annealing from the identity order (options `LS`, `LK`, `GD`, `Random`), cached at the first evaluation until `reset_ids_shuffle` is called before test. Here `SA` is a seeded annealing over swaps starting from the greedy path with an `atsp_steps` budget and `GD` is nearest-neighbour; `LS`, `LK`, `Random` are not provided. The order is recomputed whenever the cost state changes (including a loaded checkpoint), so validation and test use the latest costs without a manual reset.
- **Patching.** The paper writes ceil(L/s) patches; official code and this model use `unfold` without padding, and the official head width `d_model * seq_len // patch_len` only matches when `stride == patch_len`, whereas the head here uses the true patch count.
- **Initialisation.** Reference Mamba step-size initialisation and the `1/sqrt(layers)` residual-projection scaling are reproduced; patch embedding and head keep PyTorch defaults, as officially.
- **Runner.** Optimiser, schedule, early stopping and mixed precision are runner configuration; the official seed 3047 is not modelled. `training_objective` is unsupported under `DataParallel`.

## Citation

```bibtex
@article{cai2026mambats,
  title   = {MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting},
  author  = {Xiuding Cai and Xueyao Wang and Yaoyao Zhu and Yu Yao},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {arXiv:2405.16440},
  url     = {https://arxiv.org/abs/2405.16440}
}
```

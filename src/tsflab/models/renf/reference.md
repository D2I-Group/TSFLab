# ReNF — reference

## Differences in detail

- Implementation: independent rewrite from Sections 2.2-2.4 (Definition 2.3, Eqs. 3-4, Fig. 3) of the paper after reading the pinned official code (`Luoauoa/ReNF`, revision `c232ec5998ea82c29e7a22cfc0f41828261aa5e4`, MIT): `models/ReNF_alpha.py`, `models/ReNF_beta.py`, `layers/RevIN.py`, `exp/exp_main.py`, `run_longExp.py`, and `scripts/*.sh`. Nothing was copied or imported.
- Horizon split: N equal segments (`num_blocks`, official `--d_layers`); every block sees all previous sub-forecasts (input width `T + (H/N) * k(k+1)/2`).
- First block: `Drop(0.1) -> Linear(T, d_ff) -> Linear(d_ff, d_ff, no bias) -> GELU -> Drop -> head`.
- Alpha blocks: `LayerNorm -> Drop(min(0.3(k+1), 0.6)) -> Linear -> Linear -> Drop -> GELU (identity in the last block) -> head`, with detached feedback and an affine RevIN.
- Beta blocks: `Norm -> Drop(max(0.1 * 0.8^k, 0.05)) -> Linear` plus the dropped previous representation, `body_layers` (official `--n_block`) Linear-Drop layers, GELU, two dropped skip terms, and the head, with a non-affine RevIN. The beta pre-normalization is LayerNorm over the block input (`--norm_name layer`) or a non-affine normalization of each time position across variates (`--norm_name instance`, the run-script default).
- Position table: a learnable `[T, V]` parameter initialised uniformly in `[-0.02, 0.02]` (catalog `positional_encoding` kind `zeros`).
- Loss: scale gamma = 20; the frequency term is the mean modulus of the rFFT difference.
- Preset defaults follow the official ETTm1 horizon-96 script (beta, N = 3, `d_ff = 2048`, dropout 0.5, LayerNorm, position table on, `alpha_freq = 0.2`).
- Beta construction: at the pinned revision `ReNF_beta.Model` passes `n_b` and `norm` to a `Decoder` whose signature no longer accepts them (an April 2026 refactor), so the beta variant does not construct there; this entry forwards them to the blocks as the earlier `CapDecoder` did (revision `6bad210`).
- Not implemented: the trainer-level exponential moving average of weights (shadow model used for validation, early stopping, and the final checkpoint, decay 0.99-0.999), the official StepLR schedule, gradient clipping at 10, and the oracle "optimal" post-combination, which selects per point with the ground truth and is analysis-only.
- Normalization: the catalog `revin` uses `sqrt(var + eps)` while the official RevIN divides by `std + eps` and multiplies back by `std` (difference of order `eps`).
- Block-wise supervision is the declared `training_objective`; validation and test use the configured criterion on the final forecast.
- Checked behaviour: concatenated input widths and horizons, dense feedback (block k sees the window and all earlier sub-forecasts), alpha detaching and linear last block, the beta block equation with latent skips, the across-variate normalization, the position table and RevIN wrapping, Eq. (4) against a hand computation including the frequency term, the training objective (including `MS` channel selection), gradients, the shape constraints, and the strict parameter schema. Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

ReNF: Rethinking the Design of Neural Long-Term Time Series Forecasters, ICML 2026 (arXiv 2509.25914, 2025-09).

## Citation

```bibtex
@inproceedings{lu2026renf,
  title     = {ReNF: Rethinking the Design of Neural Long-Term Time Series Forecasters},
  author    = {Lu, Yihang and Meng, Xianwei and Chen, Enhong},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2509.25914}
}
```

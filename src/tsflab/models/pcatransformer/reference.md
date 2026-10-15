# PCATransformer — reference

## Paper

- **Title**: Transformer Multivariate Forecasting: Less is More?
- **Venue**: AI4TS workshop at AAAI 2024 (arXiv 2401.00230, 2023-12, v2 2024-03)

## Implementation mapping

- Independent local rewrite from the paper (framework of Fig. 2, problem formulation and PCA component study) after reading the pinned official code (`jingjing-unilu/PCA_Transformer`, revision `524a3b79bba9fe153491e08d46f4687ff964b1d2`, MIT): `pca_tran/data_provider/data_loader.py`, `pca_tran/models/Transformer.py`, `pca_tran/exp/exp_long_term_forecasting.py` and `pca_tran/scripts/long_term_forecast/test2cp_traffic.sh`. Nothing was copied or imported.
- Resolved from the official code: PCA runs after per-column standardization on every column except the target (the last column), and the reduced series is `[principal scores, target]` (the loader's `n_components` is the reduced channel count minus one); runs use `features = MS`, `c_out = 1`, `label_len = 48`, and the Time-Series-Library defaults (`d_model = 512`, `n_heads = 8`, `e_layers = 2`, `d_layers = 1`, `d_ff = 2048`, `dropout = 0.1`, GELU, `timeF` marks). The paper selects the number of components per dataset (Tables 3 and 6); the preset uses 2.
- The zero forecast horizon is appended in the reduced space, as the official loader transforms the data before the decoder input is built.
- Task contract: the target is the last channel and the output is `[batch, pred_len, 1]`; construction with any feature mode other than `MS` raises `ValueError`. The decoder label window is taken from `x_dec` (its first `x_dec.shape[1] - pred_len` steps) or, without `x_dec`, from the last `label_len` history steps.

## Differences in detail

1. The official loader fits the PCA on the whole standardized series, including the validation and test rows; here it is fitted on the training split only (`training_setup`), which removes that leakage.
2. The PCA is deterministic SVD PCA with a fixed sign convention; scikit-learn may use a randomized solver for large inputs and orients axes differently, which changes only the signs of the scores.
3. Before `training_setup` runs (for example in a bare forward call) the projection is a placeholder that keeps the first `n_components` covariates.
4. Only the vanilla Transformer member of the paper's six backbones (also PatchTST, Crossformer, Autoformer, Non-stationary Transformer, iTransformer) is provided.
5. The official repository does not include its `run.py` entry point; runner defaults are those of the Time-Series-Library it builds on. Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{xu2023transformer,
  title     = {Transformer Multivariate Forecasting: Less is More?},
  author    = {Xu, Jingjing and Wu, Caesar and Li, Yuan-Fang and Bouvry, Pascal},
  booktitle = {AI4TS@AAAI'24},
  year      = {2024}
}
```

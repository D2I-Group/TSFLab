# ReCycle — reference

## Differences in detail

- Implementation: independent rewrite of Sec. V (PCC, RHP, residual encoder-decoder; Fig. 2) and Sec. VI-A-b of arXiv 2405.03429. The official repository (`Helmholtz-AI-Energy/ReCycle`, revision `654d0600c98977e345d5ba90dfa3a12957049627`, MIT) was read to resolve omissions: `ReCycle/models/torch_modules/oneshot_transformer.py`, `ReCycle/models/torch_modules/model.py`, `ReCycle/data/rhp_datasets.py`, `ReCycle/data/dataset.py`, `ReCycle/data/embeddings.py`, `ReCycle/specs/model_specs.py`, `ReCycle/specs/spec_factory.py` and `ReCycle/__main__.py`; nothing was copied or imported.
- Model width is `d_model * n_heads`; encoder and decoder inputs are `Dropout -> Linear(cycle_len + 9, width)` (identity when the sizes match) and the output head is `Dropout -> Linear(width, cycle_len)`.
- The backbone is `torch.nn.Transformer` (its Xavier initialization, post-norm layers with final encoder and decoder LayerNorms, ReLU, no positional encoding, no masks); the decoder is fed the forecast RHP (`meta_token = True`).
- Day metadata is the weekday one-hot plus holiday flags for the day and the next day.
- The loss is MAE between the reconstructed forecast (residual plus RHP) and the target.
- The default RHP variant is `LooseTypeLastRHPDataset` with `rhp_cycles = 3`; profiles of a type that has not occurred are zero.
- Cycle tokens: catalog windows slide by one step, so a cycle token is the `cycle_len` steps counted back from the end of the window and its day type is the weekday of its first step; with windows that start at midnight this is the paper's fixed cycle start, otherwise a token straddles two calendar dates.
- Holidays are not available in the catalog marks, so both holiday flags are zero and only Sundays form the third type.
- Weekdays come from the raw encoder marks (and decoder marks when given, otherwise the weekday advances by one per cycle); without marks every cycle is treated as a Monday.
- Profiles are computed inside the forecast window from its own history, so `seq_len` must contain enough cycles of each type (the paper's 21 days hold three of each); the official pipeline computes them over the whole series.
- The series is normalized by the catalog loader instead of the paper's min-max scaling to `[0, 1]`. Training settings (Adam without schedule, MAE) are run settings. Reported benchmark numbers are not reproduction claims of this implementation.
- Checked behaviour: day-type categories and metadata layout, the RHP table against a brute-force average of the last `k` same-type cycles (including the zero profile before a type occurs), both `rhp_mode` variants, PCC token layout and channel independence, the forecast as the transformer residual added to the forecast RHP (zero residual falls back to the profile forecast), weekday extrapolation without decoder marks, gradients, and the strict parameter schema.

## Paper

ReCycle: Fast and Efficient Long Time Series Forecasting with Residual Cyclic Transformers, IEEE CAI 2024 (arXiv 2405.03429).

## Citation

```bibtex
@inproceedings{weyrauch2024recycle,
  title     = {ReCycle: Fast and Efficient Long Time Series Forecasting with Residual Cyclic Transformers},
  author    = {Weyrauch, Arvid and Steens, Thomas and Taubert, Oskar and Hanke, Benedikt and Eqbal, Aslan and G{\"o}tz, Ewa and Streit, Achim and G{\"o}tz, Markus and Debus, Charlotte},
  booktitle = {IEEE Conference on Artificial Intelligence (CAI)},
  year      = {2024}
}
```

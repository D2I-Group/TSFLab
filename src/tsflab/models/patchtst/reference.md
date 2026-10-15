# PatchTST — reference

## Paper

- **Title**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **Venue**: ICLR 2023 (arXiv 2211.14730, 2022-11)
- **Abstract (shortened)**: Segments each series into subseries-level patches used as Transformer tokens, and uses channel independence: every channel is a univariate series sharing the same embedding and Transformer weights. Patching keeps local semantics, cuts attention cost quadratically for a fixed lookback, and lets the model attend longer history. The paper also reports strong masked self-supervised pretraining and cross-dataset transfer.

## Differences in detail

**Clean-room implementation: confirmed.** `patchify` maps the paper's
overlapping patch equation; folding channels into the batch axis enforces
channel independence and shared Transformer weights; the flatten head maps all
patch tokens to the horizon. Inputs are `[B, context_window, enc_in]`; outputs
are `[B, target_window, enc_in]`; marks/decoder inputs are accepted and ignored.
Self-supervised pretraining, transfer learning, residual-attention accumulation,
and checkpoint or published-metric reference comparison are not included.

**Component-extraction note.** This model does not consume the cataloged
`patchtst.PatchTSTBackbone` (used by `quantile_patchtst`). The two diverge
materially: the backbone's `"zeros"` positional-encoding mode draws
`uniform_(-0.02, 0.02)` noise while this model's `"zeros"` mode is a literal
zero tensor; the backbone's encoder is `tst_transformer.TSTEncoder`
(`nn.TransformerEncoder`/`nn.TransformerEncoderLayer`, fused in-projection
attention) while this model uses a bespoke `PatchEncoderLayer`; and the
backbone's `FlattenForecastHead` applies dropout after the linear head while
this model applies `head_dropout` before it. Migrating would change both
parameter initialization and `state_dict()` keys, so per the rule that keeps
checkpoint-loading attribute names, this model stays on its own local
backbone.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/NieNSK23,
  author    = {Yuqi Nie and Nam H. Nguyen and Phanwadee Sinthong and Jayant Kalagnanam},
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  booktitle = {The Eleventh International Conference on Learning Representations, {ICLR} 2023},
  publisher = {OpenReview.net},
  year      = {2023},
  url       = {https://openreview.net/forum?id=Jbdc0vTOcol}
}
```

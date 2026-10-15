# CLoRA — reference

## Implementation mapping

- Independent implementation; no official code was copied. Checked the paper (Sec. 3, Eq. 3-7, Table 1,
  the rank study) and, at the pinned revision, `models/iTransformer.py`, `models/RMLP.py`,
  `models/TSMixer.py`, `run.py` and `scripts/long_term_forecast/*`.
- Normalization is `revin`; TokenEmbedding is `embed.DataEmbedding_inverted(seq_len, D)`; Eq. (5)-(6) are
  `ChannelLowRankAdapter.adapters` and `.forward`; Eq. (7) is `Model.adapted_tokens`; ChannelMixing is the
  iTransformer encoder (`transformer_encdec` post-norm layers with non-causal
  `self_attention_family.FullAttention` and a final LayerNorm) or the RMLP residual MLP; Projection is
  `Model.projection`. The channel low-rank adapter is model-local (it is the method itself).
- `forward` reads only `x_enc` `[B, seq_len, enc_in]`; marks and decoder inputs are ignored, as in the
  official runs (the embedding is called without marks).

## Differences in detail

- **Backbones.** The paper evaluates seven backbones (Informer, Autoformer, FEDformer, FreTS, RMLP, TSMixer,
  iTransformer). This entry provides the two inverted-token backbones whose integration is a direct instance
  of Eq. (3): `itransformer` (default; the first Table 1 column) and `rmlp`. The others need per-architecture
  inverted rewrites and are not provided.
- **Naming.** Official `--rank` is `rank` (`r`) and `--node_dim` is `adaptation_dim` (`d`); as officially, the
  token embedding width is `d_model - adaptation_dim`, so the backbone width stays `d_model`. `n_heads`,
  `e_layers`, `d_ff` and `activation` apply to the iTransformer backbone only.
- **Defaults.** `rank=16`, `adaptation_dim=32`, `d_model=d_ff=512` follow the official Weather/ECL iTransformer
  scripts; the ETTh1 iTransformer scripts use `d_model=d_ff=128`, `rank=32`, `adaptation_dim=64`. The pinned
  `run.py` does not declare `--rank`/`--node_dim`, so values are taken from the scripts.
- **Normalization.** Official iTransformer normalization leaves the standard deviation attached to the graph;
  `revin` detaches it. Forward values and parameter gradients are identical (the statistics depend only on
  data). RMLP uses affine `revin`, as officially.
- **Initialization.** Adapter factors use Xavier-uniform over the `[C, D, r]` tensor, as in the official code;
  the paper does not specify it.
- **License.** No license at the pinned revision (an MIT `LICENSE` was deleted in commit `2969a28`); recorded
  as `NOASSERTION`.

## Citation

```bibtex
@inproceedings{nie2024clora,
  author    = {Tong Nie and Yuewen Mei and Guoyang Qin and Jian Sun and Wei Ma},
  title     = {Channel-Aware Low-Rank Adaptation in Time Series Forecasting},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24)},
  year      = {2024},
  doi       = {10.1145/3627673.3679884}
}
```

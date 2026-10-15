# FPPformer — reference

## Differences in detail

- Sources read: `FPPformer/FPPformer.py`, `FPPformer/Modules.py`,
  `FPPformer/embed.py`, `utils/RevIN.py`, `utils/masking.py`,
  `exp/exp_model.py`, `scripts/Main.sh`.
- Resolved from the official code (the paper leaves these open):
  - every attention is single-head with one shared key/value projection and an
    output projection;
  - residual + post-LayerNorm order in each stage: element attention residual,
    flatten, norm; patch attention residual, norm; GELU feed-forward of width
    `4 * P * D` with residual, norm;
  - value embedding is a per-value affine map (official kernel-1 `Conv2d` from
    one channel, here `nn.Linear(1, d_model)`) plus the sinusoidal table
    (`embed.PositionalEmbedding`, identical formula);
  - decoder queries are the position embeddings of steps `L .. L+H-1`; when the
    horizon is not a multiple of the top patch size the latest embedded inputs
    are prepended and the extra outputs dropped;
  - linear branch: `Linear(D, 1)` then `L -> max(2L, 2H) -> H` with no
    activation;
  - RevIN without affine (`revin` with `affine=False`: mean and biased variance
    with `eps` inside the root, detached statistics);
  - defaults `patch_size = 6`, `num_stages = 3`, `d_model = 32`, dropout 0.1
    (`scripts/Main.sh`, Section V-B).
- The official last-stage reshape requires an even patch count and even base
  patch size; here only `seq_len` divisibility and two top-stage patches (for
  the diagonal mask) are required.
- Checked: diagonal-masked shared-key/value attention (no self weight, rows sum
  to one), encoder and decoder stage equations against hand-composed sublayers,
  merge/split reshapes, bottom-up patch sizes and top-down lateral pairing,
  decoder queries, the output inside RevIN, channel independence. Reported
  benchmark numbers are not reproduction claims of this implementation.

## Citation

Shen, L., Wei, Y., Wang, Y., Qiu, H. "Take an Irregular Route: Enhance the Decoder of Time-Series Forecasting Transformer." IEEE Internet of Things Journal 11(8), 14344-14356 (2024). doi:10.1109/JIOT.2023.3341099.

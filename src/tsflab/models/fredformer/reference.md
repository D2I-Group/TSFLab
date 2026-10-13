# Fredformer reference alignment

Official source: https://github.com/chenzRG/Fredformer at
`fa64775ea1012e313cbe30fe2c9b7e493a798aae`; inspected `models/Fredformer.py`,
`layers/Fredformer_backbone.py`, `layers/cross_Transformer.py`, and `layers/RevIN.py`.
No license file at that revision; external source is not distributed in TSFLab.

The standard forecast path uses full FFT (not rFFT), unpadded frequency patches,
concatenated real/imaginary features, channel-axis pre-norm attention, two separate
spectral projections, three residual ReLU layers per shared flatten head, inverse
FFT and a learned real/imaginary output projection. There is no explicit RMS-band
equalization or learned energy gate in that official path.

TSFLab parameter names map as follows: band_width=patch_len, model_width=cf_dim=d_model,
depth=cf_depth, heads=cf_heads, head_dim=cf_head_dim, feedforward=cf_mlp,
dropout=cf_drop. This exposes the common tied embedding/output width configuration;
independent official cf_dim/d_model widths are not exposed. The preset is a controlled
configuration, not a claim to reproduce every dataset-specific official recipe.

The shared head in the reference declares but does not apply head_dropout. TSFLab
preserves that behavior. Unused ablation layers are excluded from parameter counts.
Compare active trainable parameter counts separately from all allocated reference
parameters. The Nystrom and individual-head variants are not implemented.

Numerical validation must use matched active weights, output and input-gradient
comparisons, including batch/channel singleton inputs and the benchmark 168/24
lengths. Admission alone is not evidence of official numerical parity.

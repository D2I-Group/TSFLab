# inception_block — reference

## Origin and granularity

The operator is TimesNet's parameter-efficient inception block (Wu et al., ICLR
2023, "TimesNet: Temporal 2D-Variation Modeling for General Time Series
Analysis"; `Inception_Block_V1` in `layers/Conv_Blocks.py` of Time-Series-Library),
a simplification of the multi-branch Inception module of Szegedy et al. (CVPR
2015, "Going Deeper with Convolutions") to same-padded square kernels whose
outputs are averaged. Before extraction four models kept a local copy: `dragon`
(DRAGON, "Multivariate de Bruijn Graphs: A Symbolic Graph Framework for Time
Series Forecasting", ICML 2025 workshop; its downstream TimesNet), `times2d`
(Times2D, AAAI 2025; the derivative-heatmap convolutions), `wftnet` (WFTNet,
ICASSP 2024; the Time-Frequency Inception Block shared by both branches), and
`timesnet` (the clean-room TimesNet). The component is cut at one block; the
GELU sandwich, the 1D-to-2D folding and the branch aggregation stay model-local.

## Invariants and equivalence evidence

- At extraction, frozen copies of the pre-extraction local blocks
  (`dragon.InceptionBlock`, `times2d.InceptionBlock2d` and
  `wftnet.InceptionBlock2D`, which were identical, and `timesnet.Inception2D`) were
  compared from the same seed: identical `state_dict()` keys and values, outputs
  and input and parameter gradients for the two-block GELU sandwich. A contract
  check also covered kernel sizes and padding, the state-dict keys, the branch
  average and gradient flow, and that `init_weight=False` reproduces plain
  `Conv2d` construction.
- A `times2d` model-level check (pre-consolidation suite) confirmed the branch
  average through the `times2d` import.
- The dragon, times2d and wftnet migrations are verbatim moves (same code, the
  attribute name `kernels`, the same registration order and defaults), equivalent
  by construction; timesnet uses the explicit option `init_weight=False`. No stored
  reference values: the comparison was against the frozen inline copies. The
  frozen-copy test passed in the full suite run of 2026-10-03 before the test suite
  was consolidated.

## Variants and options

- `init_weight=True` (default): Kaiming-normal fan-out weights and zero bias
  (`dragon`, `times2d`, `wftnet`, matching the official Time-Series-Library code).
- `init_weight=False`: PyTorch default `Conv2d` initialization (`timesnet`, whose
  clean-room implementation never re-initialized; recorded in its card).
- No option for rectangular, dilated or grouped kernels, concatenation instead
  of averaging, or the `Inception_Block_V2` (1 x k and k x 1 strip kernels) variant.

## Related components

- `dominant_periods`: finds the periods by which `timesnet`, `dragon` and
  `wftnet` fold the series into the 2D images this block convolves.
- `gated_dilated_conv`: causal 1D dilated convolutions with a gated activation,
  for sequence (not image) inputs.
- `fft_extrapolation_conv`: convolution evaluated in the frequency domain.

---
name: "HL"
description: "Historical Last: parameter-free persistence baseline repeating each node's last observed value over the whole horizon. Use as a lower-bound reference on any graph or multivariate benchmark; not as a forecaster when dynamics move away from the last value."
---

# HL

## Idea

- Historical Last (HL) returns `x_enc[:, -1:, :]` expanded across `pred_len`; no learned parameters, graph or covariates are used.
- Serves as the lower-bound reference for graph and node-structured benchmarks: any learned model should beat it, especially at longer horizons where dynamics diverge from the last observation.
- Kept model-local because the `last_value_center` component assumes a head between subtract and restore, which HL lacks.

## When to use

- As a sanity floor for spatiotemporal and multivariate benchmarks; needs no training data and no compute.
- Not as a forecaster in its own right: it cannot follow trend, seasonality or any change after the last observation.

## Configure

- `enc_in`: number of nodes or channels; used only to check the input shape.

No other parameters.

## Differences

- In-repository persistence baseline with no associated paper or canonical BibTeX entry; clean-room formula, nothing copied from the reference repository (`PoorOtterBob/CauAir`, no license file, `NOASSERTION`).
- No paper or checkpoint reference comparison is claimed.
- The forecast is exactly the detached last observed step broadcast across the horizon; with no head between a subtract and an add, the `last_value_center` contract does not apply.

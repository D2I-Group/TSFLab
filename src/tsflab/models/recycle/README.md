---
name: "ReCycle"
description: "Residual cyclic Transformer: compresses each day into one token and learns only the residual against recent same-weekday-type profiles, decoding all forecast days in one pass. Use for long-input hourly load or traffic data with a daily cycle and weekday effects; not for aperiodic series."
---

# ReCycle

## Idea

- Primary cycle compression (Sec. V-A): each channel's history becomes `seq_len / cycle_len` tokens of `cycle_len` values (one per day), so attention compares whole daily profiles and the sequence shrinks by the cycle length; channels are independent series sharing one model.
- Recent historic profiles (RHP, Sec. V-B): for each day type (Mon-Fri, Sat, Sun) the mean of the last `rhp_cycles` cycles of that type; `rhp_mode = "last"` (official default) uses the profiles available after the last historic cycle, `"causal"` gives each historic cycle only earlier cycles.
- Encoder input is `cycle - RHP` plus 9 day-metadata features (weekday one-hot, two holiday flags); decoder input is the forecast-cycle RHP with its metadata (Fig. 2).
- `nn.Transformer` (post-norm, no masks, single-pass decoding) gives one residual cycle per forecast cycle, added to the forecast RHP (Sec. V-C); a zero output falls back to the profile forecast.

## When to use

- Long inputs with a dominant daily cycle and weekday/weekend differences (electricity demand, traffic): compression makes long histories cheap, and the profile baseline captures the regular pattern.
- Timestamps must carry the weekday (raw encoder marks); without marks every cycle is treated as a Monday.
- Not for aperiodic data or cycles that do not align with the sampling grid; the lookback must hold several cycles of each day type.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `cycle_len` follows the dataset period: steps per primary cycle (24 for hourly daily cycles); `seq_len` and `pred_len` must be whole multiples of it.
- `rhp_cycles` follows `seq_len`: the lookback must contain enough cycles of each day type (the paper's 21 days hold three of each).

Other hyperparameters: preset defaults in `configs/models/ReCycle.toml`; tune generically.

## Differences

Independent rewrite of Sec. V and VI-A-b after reading the official MIT repository (`654d060`); nothing copied.

- Cycle tokens count back from the end of each sliding window; unless windows start at midnight, a token straddles two calendar dates.
- Holidays are not in the catalog marks, so both holiday flags are zero and only Sundays form the third type.
- Profiles are computed inside each window from its own history (official: over the whole series).
- Catalog loader normalization replaces the paper's min-max scaling.

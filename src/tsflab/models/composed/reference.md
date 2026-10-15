# Composed — reference

## Slot mechanics

- The `composition` field in `card.toml` lists the selectable options per slot (alternatives joined by `+`),
  not options used together. The sixth slot, the loss, stays in `training.loss`.
- `tsflab.models._slots.registry` is the executable option table: `OPTIONS`, `canonical` (aliases such as
  `component:revin`) and `check_assignment`, which returns every reason a combination cannot run;
  `ModelParameterConfig` rejects those combinations at config validation (for example `mixer_block` needs
  `channel = "mixing"`, and `channel = "mixing"` needs `mixer_block`).
- `tsflab.models._slots.adapters.Pipeline` runs `normalization.norm`, then `decomposition.split` (one branch,
  or residual and trend for `series_decomposition`), one temporal encoder per branch producing
  `z [B, C, D, P]`, concatenation of the branches along `P`, the `flatten_forecast_head`, and
  `normalization.denorm`. `temporal = "linear"` has no encoder: the head's linear map over the flattened
  history is the temporal model.
- `channel` sets weight sharing: `independent` shares weights across channels, `individual` gives per-channel
  weights in the head and in `channel_wise_linear`, `mixing` is only provided by `mixer_block`'s feature MLP.

## Executable and rejected options

- `normalization`: `none`, `revin` (non-affine), `last_value_center`. `decomposition`: `none`,
  `series_decomposition` (`kernel_size`, odd). `temporal`: `linear`, `channel_wise_linear`, `tst_transformer`
  (patched, sinusoidal positions, feed-forward width `2 * hidden`), `mamba` (patched, pre-norm residual blocks
  with `d_state=8`, `d_conv=4`, `expand=2`), `gated_dilated_conv` (causal gated convolutions, kernel 3,
  dilations `2**i`, residual), `mixer_block`. `channel`: `independent`, `individual`, `mixing`. `head`:
  `flatten_forecast_head`.
- The parameter schema forbids unknown keys and its `Literal` types list only executable options.
  `decomposition = "wavelet"` (the component has no uniform `[B, L, C]` branch interface) and the
  `quantile_head` / `gaussian_parameter_head` heads (Composed is point-output only) are rejected by field
  validators with an explanatory message; the slot registry still lists them so composition specs can be
  checked and registered with `tsf model compose <spec> --register NAME`.
- `hidden`, `n_layers`, `n_heads`, `patch_len`, `stride`, `kernel_size` and `dropout` apply only to the options
  that use them; `hidden` must be divisible by `n_heads` even when the temporal option is not a transformer.
  Schema defaults are `normalization = "none"` and `temporal = "linear"`; the preset sets `normalization = "revin"`.
- Any point loss from `training.loss` is valid; the head has no loss of its own and the model defines no
  `training_objective` or auxiliary loss.
- The spec declares no smoke config; `tsf model compose spec.toml --write-config run.toml --dataset <name> --smoke`
  writes a one-epoch CPU config.

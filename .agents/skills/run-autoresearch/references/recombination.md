# Recombine components and models

Freedom is kept, but common choices are explicit: fill structured slots first, use the
free-form slot only for what the catalog cannot express.

## Slot grid

Slots and static defaults are the `[slot.*]` tables in
`src/tsflab/data/profile_rules.toml`; `recommendations.slots` in the profile adds the
options promoted by fired rules.

| Slot | Choice | Typical options |
| --- | --- | --- |
| normalization | input stats and restoration | `component:revin`, `component:last_value_center`, none |
| decomposition | trend/seasonal or frequency split | `series_decomposition`, `periodic_query_bank`, `dominant_periods`, `wavelet`, `freq_band_moe` |
| temporal | backbone along time | `channel_wise_linear`, `patchtst`, `mixer_block`, `mamba`, `gated_dilated_conv` |
| channel | cross-channel strategy | independent (shared weights), mixing (`iTransformer`/`SOFTS`/`TQNet` idea), low-rank, graph for node data |
| head | forecast projection | `flatten_forecast_head`, `quantile_head`, `gaussian_parameter_head` |
| loss | `training.loss` | `loss:mse`, `loss:mae`, `loss:quantile`, `loss:nll_gaussian`, `loss:masked_mae` |
| free-form | one bounded new block | any role; at most 2 blocks of at most 120 lines each |

Option syntax is `component:<name>`, `model:<Name>` (donor design whose idea is borrowed;
models never import peers), or `loss:<name>`. Components must be real
(`uv run tsf catalog list --kind component`); `tsf catalog match <preset>` lists the ones
whose `fits` match the data, grouped by their card `slot`. Check shapes, axes,
normalization, masking, and residual order in the component card's Interface
(`tsf catalog show <name...>`, all chosen components in one call) before combining.

## Fill and prune from the profile

1. Start from the best baseline's slot assignment; this is the incumbent.
2. For each slot, list defaults plus options promoted by fired rules or matched by card
   `fits`. Each promoted option must cite the characteristic and hypothesis that justify it.
3. Prune: drop options that contradict a profile fact (period decomposition when seasonal
   strength is weak; channel mixing when cross-channel is weak and channels are many;
   attention backbones when `split.train` is short), exclude duplicates (two normalizers),
   and drop options the baseline panel already showed to be inert.
4. Rank the survivors by cost and expected information; screen single-slot swaps on the
   incumbent first (one factor per iteration), then combine the two best compatible
   winners. Interactions are the question only for that final combination.
5. A config-level swap needs no model code: change `training.loss`, `task.seq_len`, or
   model parameters (for example RLinear `affine`, PatchTST `revin`, CycleNet `use_revin`).
   Data-dependent parameters (CycleNet `cycle`, DLinear `kernel_size`) follow the card's
   `[data_params]` rule, not a search; generic hyperparameters (width, depth, dropout,
   learning rate) are yours to tune with a few values per iteration within the budget.
   A new component assembly needs code authorization.

## Free-form slot

Use it only when no component covers a profile-justified need. State why in the spec,
keep the block model-local, budget its lines, and check its shapes before running. Never copy code from
official repositories or peer models; extract a component later only when two consumers
match exactly (`curate-components`).

## Composition spec and dry run

Write a TOML spec (outside the catalog, for example under the round directory):

```toml
name = "SeasonalRevLinear"
summary = "RevIN and decomposition around a channel-wise linear map."
hypothesis = "strong-seasonality: period-aware split beats RLinear at equal lookback."
parents = ["RLinear", "DLinear"]
[slots]
normalization = ["component:revin"]
decomposition = ["component:series_decomposition"]
temporal = ["component:channel_wise_linear"]
loss = ["loss:mae"]
```

`uv run tsf model compose <spec.toml>` writes nothing. It verifies component, model,
and loss names, public-symbol imports, one loss, the free-form budget, and prints the
`tsf model scaffold` command and a provenance sentence. It does not prove shape
compatibility. `--write-config PATH --dataset D --enc-in N [--smoke]` emits a runnable
config for the round; `--register NAME [--dry-run]` scaffolds a model package with a
`composed` card for the winner.

## Register a winner

Only with authorization, after finalists beat the baseline panel with confirmation seeds:

1. `tsf model compose <spec.toml> --register <Name>` (or `add-model` with the scaffold
   command from the dry run; `--components` lists exactly the composed components). The
   card has `fidelity = "composed"`, no `[paper]` or `[code]`, and states in its README
   the composition, donors, hypothesis, profile facts, research round id, and material
   differences; set `fits` to the characteristics the round showed it helps with.
2. Admit it: `uv run tsf model add --name <Name> --verify` (writes `[admission]`,
   refreshes generated pages, and runs the audits).
3. Compare the registered model against the whole catalog under the same protocol before
   claiming a result; note it in the round as a `conclusion`.

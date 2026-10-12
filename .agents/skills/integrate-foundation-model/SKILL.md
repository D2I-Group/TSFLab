---
name: integrate-foundation-model
description: "Integrate a released pretrained time-series foundation model through its official package and a pinned local checkpoint. Use for zero-shot or inference-only foundation runtimes; not for ordinary paper architecture implementations."
---

# Integrate a foundation model

Models module, variant of `add-model`: expose an official pretrained runtime as one flat, inference-only catalog entry.
Do not reimplement the network, copy its source, convert checkpoints without a
demonstrated need, or bundle weights in TSFLab.

## Inputs

- Primary paper, official repository, package version, checkpoint identifier,
  pinned revision, license, input semantics, outputs, and supported horizons.
- A local checkpoint path, or a checksum-pinned `hf://` artifact to fetch explicitly.

## Steps

1. Confirm identity and license; an architecture-only local model is a different
   claim and belongs to `add-model`.
2. Check whether the official package coexists with the main environment. If
   dependencies conflict, use a compatible provider environment behind the same
   `src/tsflab/models/_foundation/` boundary; never relax core dependencies.
3. Reuse `FoundationModel` and the closest official runtime adapter. Load only
   from an explicit local path with offline mode on; no network request during
   construction, verification, or an experiment.
4. Add one ordinary `src/tsflab/models/<slug>/` entry whose factory receives
   verified local artifacts, declares `inference-only`, and exposes the canonical
   four-input interface. Do not add a provider registry or a foundation category.
5. Write the card as in `add-model` step 5: `card.toml` with `[paper]`, `[code]`
   (official package repository, revision, license), `fidelity`, `fits` (e.g.
   `low-data`, `probabilistic-output`), composition, and `[data_params]` (context
   length, patch or frequency rules); README Configure covers checkpoint facts and
   cache preparation, Differences every adapter transformation, preprocessing, and
   channel treatment. Compose upper-layer methods only through experiment configuration.
6. Check offline failure, official reference outputs, tensor axes, quantiles,
   finite values, CPU behavior where supported, and batch/sequence bounds; the
   admission contract skips training because the runtime is inference-only.

```bash
uv run tsf model artifacts <Name>
uv run tsf model add <Name> --verify   # offline, CPU, small tensors: writes [admission]
```

## Chain

- Module: Models.
- Reads: official package facts, component cards (L0) for shared interfaces.
- Produces: an inference-only card (with `[admission]`), spec, pinned artifacts, preset (same layout as `add-model`).
- Hands off to: `run-experiment` (zero-shot runs), `run-autoresearch`, `audit`.

## Success

- One flat entry with pinned artifacts, offline construction, outputs that match
  the official reference, and `[admission]` status `passed`.

## Stop and hand off

- Stop if the license, official loader, checkpoint identity, or input/output
  semantics cannot be established; an adapter alone is not a catalog model.
- Admit the entry with `tsf model add <Name> --verify` exactly as `add-model`
  step 7 does (static audits, admission contract; fix and rerun on failure).

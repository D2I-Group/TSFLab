# Implement a model locally

Write one model as local, verified code built from cataloged components where the
semantics match. A released pretrained foundation model is not rewritten; use
`integrate-foundation-model`.

1. Resolve paper omissions (tensor order, padding, initialization, defaults,
   train/eval behavior) from the paper and pinned official code. Never copy, rename,
   mechanically rewrite, import, or depend on external model source. A missing
   official license does not block an independent rewrite; record it as missing.
2. Reuse `src/tsflab/models/_components/` only after proving mathematical and runtime
   equivalence. Implement each `extract-new` operator as a component with its own
   card, tests, and `ComponentSpec`; keep paper-specific glue local.
3. Implement inside the flat `src/tsflab/models/<slug>/` package with the exact
   four-input `forward(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)`.
   Model-specific operations belong in named methods, not extra public inputs. Keep
   useful paper formulas as comments, never source-derived code or comments.
4. Keep `spec.py` to construction, a strict parameter schema (no `**kwargs` or hidden
   aliases), config, capabilities, declared components, runtime contract.
5. For pretrained weights or tokenizers, declare checksum-pinned `ModelArtifact`
   entries (`hf://`, `https://`, or `file://`), never download implicitly, and add an
   `artifact_factory(cfg, params, paths)` that uses verified local paths only.
6. Complete the card: `card.toml` facts (paper, code with `reference_sources`,
   fidelity, composition, data_params, issues) and the README sections Idea, When
   to use, Configure (data-dependent parameters only), Differences.
7. Check equations and structure against the paper and, when official code exists,
   compare against it at the pinned revision; the card's `fidelity` and Differences
   record what was compared (`reference-checked` needs official code).

```bash
uv run tsf catalog show <Name>     # L1; --depth 2 for the full card, --depth 3 for paths
uv run tsf model verify <Name>   # existing model; new entries use tsf model add --verify
uv run tsf model audit <Name>
```

Success: `[admission]` passed, the card is truthful, no peer-model import, no
unresolved defining operation. Stop rather than add a placeholder when the paper, inputs, or defining behavior are too ambiguous to implement truthfully.

# Audit one model

Check that one catalog entry's claims, code, and admission record agree.

```bash
uv run tsf catalog show <Name> --depth 2   # card facts, README, reference.md
uv run tsf model audit <Name>
```

1. Treat `card.toml` as the canonical facts. Check `[paper]`, `[code]` (url, license,
   revision, `reference_sources`) or its recorded absence, `fidelity` against what
   was actually compared, `[[issues]]` at the pinned revision, `[data_params]`
   against the preset, and that the README sections match the code.
2. Compare equations, shapes, defaults, objective, preprocessing, initialization, and
   output semantics. Confirm the code is local, the official revision was inspected
   when available, nothing was copied or imported, and every defining operation has
   a justified component decision.
   Close each defining equation against its source symbol, tensor/reduction axes,
   residual operands, branch gates and normalization scope. Distinguish a finite
   admission result from mathematical fidelity; missing residuals and accidental
   channel averaging require implementation repair, not tuning against test scores.
3. For runtime artifacts: every required asset is checksum-pinned and fetched
   explicitly, the artifact factory receives only verified local paths, offline
   absence fails before construction, and the card claims no checkpoint behavior
   beyond what verification covers.
4. For an inference-only foundation runtime: it uses the official loader through
   `src/tsflab/models/_foundation/`, never downloads implicitly, skips training, and
   says so in its card.
5. Re-run the admission contract (construction, forward and backward, finite outputs,
   active gradients, state-dict round trip, CPU, declared marks or adjacency) and read
   the `[admission]` it writes (status, commit, note):

   ```bash
   uv run tsf model verify <Name>
   uv run tsf run --smoke --model <Name>   # only when `tsf catalog show` reports a smoke_config
   ```

Report findings per layer (card facts, source facts, implementation, admission), each
traced to a file and line. Never persist status or blockers in metadata. For several
models, audit each independently and let the task own partitioning, shared-file
writes, and the final repository gate.

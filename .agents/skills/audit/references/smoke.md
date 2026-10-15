# Smoke checks

Exercise construction, a short training path, evaluation, and the declared output
shape on tiny data. Smoke checks are not admission and not a fidelity claim.

```bash
uv run tsf run --smoke --model DLinear
uv run tsf run --smoke --config configs/runs/smoke_dlinear.toml
uv run tsf run --smoke --all --jobs 8
```

A model is smoke-testable only when `tsf catalog show <Name>` reports a `smoke_config`;
for the others, rely on `tsf model verify <Name>` (the admission contract). Add focused cases for material optional
objectives or output types. Every selected config must report PASS; keep the final
diagnostic of each failure and route it to `diagnose-experiment`.

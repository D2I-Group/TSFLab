<!-- Thanks for contributing to TSFLab! Fill in the sections below. -->

## Summary

<!-- What does this PR do and why? -->

## Type of change

- [ ] New model
- [ ] New dataset
- [ ] Bug fix
- [ ] Feature / framework change
- [ ] Docs only

## How was it tested?

<!-- Paste the focused verification and smoke commands that apply. When a model
     declares a smoke config, run it for one epoch on CPU, e.g.:
       UV_TORCH_BACKEND=cpu uv run tsf run --smoke --config configs/runs/smoke_<name>.toml
     Also report `tsf model verify <Name>` and the strict model contract. -->

## Checklist

- [ ] Focused tests and every declared smoke run pass.
- [ ] New model/dataset has its spec, config, readable card, and unified verification evidence.
- [ ] Model changes pass `uv run tsf repo check --contracts strict --models <Name>` and applicable audits.
- [ ] Official source facts are pinned and cited; external model source was not copied or vendored.
- [ ] English documentation and generated catalogs are current.
- [ ] No hardcoded `+cuXXX` torch pin added to `pyproject.toml` (the build is chosen via `UV_TORCH_BACKEND`).

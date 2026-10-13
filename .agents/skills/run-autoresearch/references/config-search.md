# Bounded search over supplied candidates

When a task already names the model, axes and training protocol, use those inputs.
No catalog panel, architecture audit or paper research is needed. Standard training
learning rate is `training.optimizer.lr`; do not invent `decoder_lr` model fields.
Model-specific pretraining knobs are only those in the named parameter schema. Query model
parameters only for an unknown field. For a custom CSV set `dataset.name=custom`
and `dataset.path=FILE`; a file label is not a dataset registry key.
For StateFlow structural search, keep the full network and change one declared
`hidden_size`, `memory_size`, `chunk_width`, or `chunk_stride` value. Nondefaults
are StateFlow-derived variants; record the factor before execution, and keep
the task’s fixed encoder/decoder training budget unchanged.

1. For a native dataset, profile once with `tsf data analyze PRESET --data-path FILE
   --json --compact`; this preserves training-loader row caps and boundaries.
   Use `--path FILE --split-ratio TRAIN VAL TEST` only when training also uses
   `dataset.name=custom`. Never profile the full ETT CSV by raw ratios while
   training uses the capped native ETT loader. For custom data use:
   `tsf data analyze --path FILE --split-ratio TRAIN VAL TEST
   --freq FREQ --json --compact`. Default output has train/val facts and no test
   statistics. Record the selection rule and short falsifiable hypotheses before runs.
2. Create one round via `tsf research start --goal QUESTION --max-runs N
   --max-iterations M --policy POLICY.toml`. For an automatic GPU sweep use:

   ```toml
   [budget]
   max_runs = 10 # replace with the task's total training/evaluation claim budget
   max_wall_minutes = 10 # replace with the task's wall deadline
   auto_parallel = true
   [resources]
   threads_per_run = 4
   ```

   Do not set `max_parallel_jobs=1` for independent cells on multiple available GPUs.
   Keep exclusive GPU allocation; the runner resolves concurrency from visible cards.
   Batch creation, hypothesis notes and iteration claim in one shell call; retain
   their native artifacts rather than printing the same ledger repeatedly.
3. Use the generic direct-request interface in run-experiment/references/request.md.
   Submit independent screening cells in one sweep with `--round ID --eval-split val
   --gpus auto --jobs auto --json --compact --output results/screening.json`.
   Native returned rows contain parameters, seeds, metrics and checkpoint paths.
4. Decide once from returned validation metrics. For a declared all-horizon
   MSE/MAE improvement gate, check the matched screening run directories with
   `tsf research screen ID --baseline BASE_DIR... --candidate CAND_DIR...`.
   When failure reaches the task's stopping condition, add `--stop-on-fail`;
   this closes the round and prevents further claims. A failed screen is a
   valid no-winner outcome, not an execution error to retry. Beating another
   losing candidate does not pass the baseline gate. Submit only missing confirmation
   seeds, reusing exact matching screening records. Do not reread manifests to
   recover parameters already known from the submitted matrix.
5. Freeze the winner, record its selected run directories, then call
   `tsf run evaluate RUN_DIRECTORY... --eval-split test --round ID --gpus auto
   --jobs auto --json --compact --output results/final-test.json`. Both training
   stages are skipped; the original best checkpoint is hashed and loaded.
6. Stop on the declared budget or no improvement. A compact completion artifact
   can cite native screening, confirmation and frozen-test files plus the selection,
   seed mean/spread, run count and stopping reason. Do not duplicate every manifest,
   full profile and ledger into a second verbose report. Do not invent timings.

Use blocking calls and wait on the original process; inspect only actual failures.
This reference plus the direct-request reference suffice for supplied candidates;
do not load general retrieval, source-audit or additional workflow references.
Use existing entrypoints and declared parameter names; do not guess alternate paths.
The runner validates matrices internally, so an extra preview is unnecessary on
an already authorized run host. Model implementations and data loaders remain the
same as ordinary experiments. One candidate test result is never selection evidence.

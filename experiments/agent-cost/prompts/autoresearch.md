Find a new forecasting method that beats {target_method} on {dataset}.

The current directory is your working directory. If it contains a codebase,
build on it and follow its structure, naming, and conventions instead of
starting from scratch. If it is empty, write everything yourself.

Raw data files are in {data_dir} (read-only). One GPU is available to you: the one
already selected by CUDA_VISIBLE_DEVICES. Use only that GPU and do not change
CUDA_VISIBLE_DEVICES.
Target: {target_method}, the best method on the current leaderboard for
{dataset}. First run it under the codebase's protocol ({split}) at prediction
lengths {pred_lens} and record its validation MSE (method name
"{target_method}", one row per seed) in results/results.json; that is the
target to beat.

Rules:
- Work without asking questions. Record every hypothesis, decision, and run in
  results/ledger.md.
- Select candidates on validation data only. Confirm your final method over
  three seeds. Read test metrics once, only for the final method.
- Explicitly set evaluation.split="val" in EVERY baseline, screening and
  confirmation config, including runs outside a research round. Inspect the
  resolved config before launching; never rely on the default split. Only a
  frozen final method may use evaluation.split="test".
- Evaluate the frozen final checkpoints without retraining. In TSFLab use
  `tsf run evaluate <successful-run-directory>... --eval-split test`; elsewhere
  load the saved checkpoint in a minimal evaluation adapter. Never select by test.
- The TSFLab harness enforces validation as the CLI default. For the frozen
  final method only, invoke checkpoint-only evaluation with `--eval-split test` explicitly.
- Batch independent candidates into one sweep. Wait for completion with
  `tsf run wait <sweep> --timeout 1800` when available, or a blocking local
  process wait in other environments. Do not repeatedly inspect logs, process
  lists or GPU status while a run is healthy; inspect logs once on failure.
- Keep one session-wide budget and one research round. Stop after two
  consecutive search iterations without validation improvement. Do not reset
  the budget by starting another round. A report with no winner is a valid
  stopping outcome; do not fabricate a final method or test results.
- At the start of each stage, run `stage <name>` with one of: collect, read,
  hypothesize, implement, search, confirm, report. Run `stage done` when you
  finish.
- Stop when you have a confirmed method that beats the target on validation,
  or when you reach the budget of {budget}.
- Write results/results.json as a JSON list of objects
  {{"method": str, "seed": int, "pred_len": int, "split": "val"|"test",
  "mse": float, "mae": float}}, and results/report.md describing the final
  method.

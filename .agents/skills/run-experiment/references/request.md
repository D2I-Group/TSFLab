# Direct model request

For a task with settled scientific settings, call the named preset directly.
`--set KEY=VALUE` accepts JSON numbers/booleans/lists or unquoted scalar strings;
`--sweep KEY=LIST` expands a Cartesian matrix. These use the same config loader,
parameter schemas, readiness checks, budgets, manifests and resume as TOML runs.
Every expanded scientific snapshot and axis is saved; a hand-written config is
unnecessary. Named requests and positional config files are mutually exclusive.

```bash
uv run tsf run --model MODEL \
  --set dataset.name=custom --set dataset.path=./DATA.csv \
  --set dataset.params.target=TARGET --set dataset.params.scale=true \
  --set 'dataset.params.split_ratio=[0.6,0.2,0.2]' \
  --set task.seq_len=96 --set task.label_len=0 --set task.features=M \
  --set task.inverse=false --set experiment.runtime.device=cuda \
  --set experiment.runtime.num_workers=0 \
  --set training.epochs=2 --set training.batch_size=64 \
  --set training.drop_last=true --set training.patience=2 \
  --set training.optimizer.name=Adam --set training.optimizer.lr=0.001 \
  --set training.optimizer.weight_decay=0 --set training.optimizer.lradj=constant \
  --set evaluation.split=val --set 'evaluation.metrics=["mse","mae"]' \
  --sweep 'task.pred_len=[24,48]' --sweep 'experiment.random_seed=[0,1]' \
  --gpus auto --jobs auto --json --compact --output results/completion.json
```

Replace `dataset.name=custom` with the task's named dataset when its native
row cap, split or window rules are required; a CSV path does not override them.
Replace placeholders and settings from the task; do not change its experiment
budget to fit this example. Add only declared model-specific fields using
`--set model.params.FIELD=VALUE`; query `tsf catalog show MODEL --params` if needed.
Fixed architecture sizes stay at preset values. Unknown keys fail before compute.
If the named schema declares `enc_in`, add `--set model.params.enc_in=CHANNELS`
using the actual input channels (one for `S`, all series columns for `M`/`MS`).
For schemas without it, such as TiDE, omit it; `decoder_output_dim` is not a
substitute for input channels. An ETT preset's seven channels are not universal.
Align other channel fields only when the named schema requires them.
CPU thread limits use execution policy `resources.threads_per_run` or OMP/MKL/
OpenBLAS environment variables; never invent `experiment.runtime.num_threads`.
Batch size counts complete multivariate windows before channel flattening.
Training `drop_last` never drops validation/test samples. No source inspection,
standalone preview or reformatting of returned JSON is needed for a settled task.

For benchmark profiling set `--set evaluation.enable_profile=true` (not an
experiment field). Native completion rows include `trainable_parameters`,
`train_time_sec` (encoder plus decoder), `test_time_sec` (configured eval split),
and `training_peak_cuda_mib` (allocated CUDA peak from construction through
training and its validation, before final inference; null on CPU/eval-only).
Profile peak fields describe profiling, not the training phase.

For a frozen finalist, evaluate saved successful run directories without training:
`tsf run evaluate RUN_DIRECTORY... --eval-split test --gpus auto --jobs auto
--json --compact --output results/final-test.json`. This derives each config from
its saved snapshot, hashes the best checkpoint, skips both training stages, and
saves separate evaluation records. Do not change the model or preprocessing.

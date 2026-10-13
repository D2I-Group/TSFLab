# Minimal run or sweep config

When task settings are already specified, edit only these fields and the named
model preset. Model architecture checks belong to admission or an explicit audit.
Extend paths relative to the new config; data/work paths are relative to execution.
This example lives in `configs/runs/`. Replace placeholders; omitted fields keep
preset defaults. Query `tsf catalog show MODEL --params` for declared model fields;
do not infer names from architecture descriptions. Request only needed metrics.
CPU thread caps belong to execution policy `[resources] threads_per_run = 4`
or OMP/MKL/OpenBLAS environment variables, never `experiment.runtime.num_threads`.
For a panel with different model schemas, write one minimal config per model and
submit them together: `tsf run A.toml B.toml ... --gpus auto --jobs auto --json
--compact --output results/panel.json`. Common training/data settings may live in
a shared base. Keep model-specific fields in their own config: `c_out` is not
universal, and TiDE uses `decoder_output_dim` rather than `enc_in`. Do not broadcast
the union of model fields through one shared `[model.params]` section.

```toml
extends = ["../base.toml", "../models/MODEL.toml"]
[experiment]
work_dir = "./work_dirs"
[experiment.runtime]
device = "cuda"
num_workers = 0
[dataset]
name = "custom"
path = "./DATA.csv"
[dataset.params]
target = "TARGET_COLUMN"
scale = true
split_ratio = [0.6, 0.2, 0.2]
[task]
seq_len = 96
label_len = 0
features = "M"
inverse = false
[training]
epochs = 2
batch_size = 64
drop_last = true
patience = 2
[training.optimizer]
name = "Adam"
lr = 0.001
weight_decay = 0.0
lradj = "constant"
[evaluation]
split = "val"
metrics = ["mse", "mae"]
[model.params]
enc_in = 7
[sweep.task]
pred_len = [96, 192]
[sweep.experiment]
random_seed = [42, 43]
```

Additional model-specific fields use `[model.params]`, as in its preset. Do not
copy training, device, or data fields into model parameters. `run` validates the
entire matrix before any cell starts; its saved JSON is already the collection
artifact. Configured epochs are budgets, and early stopping can finish sooner.

Batch size counts multivariate windows, before a channel-independent model flattens
channels. `training.drop_last` affects training only; validation/test always keep
the final partial batch. Its default `None` retains historical automatic behavior.

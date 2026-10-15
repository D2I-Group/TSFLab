# SAMformer — reference

## Differences in detail

Paper and official code (pinned revision `71f10ea`, `samformer_pytorch/samformer/`) were checked; the local module is an independent rewrite. The architecture matches the official `SAMFormerArchitecture`: RevIN, single-head channel-wise attention with `Q, K: seq_len -> hid_dim`, `V: seq_len -> seq_len`, scale `1/sqrt(hid_dim)`, residual add, one shared `seq_len -> pred_len` linear forecaster, RevIN denormalization.

- SAM is not an optimizer wrapper here. The official `SAM` optimizer runs `first_step` (ascent to `w + rho g/||g||`) and `second_step` (restore `w`, then step the base optimizer). TSFLab exposes `spec.training_objective`, which evaluates the criterion at the perturbed weights through `torch.func.functional_call` (first-order, perturbation detached), so the run's ordinary optimizer applies the same update at the same cost of two forward and backward passes.
- Gradient clipping, schedulers, weight decay, mixed precision, and the optimizer itself come from the run config, not from the official defaults (Adam, `lr=1e-3`, `weight_decay=1e-5`, `rho=0.5`).
- `training_objective` is used only in training; the validation and test loss and `forward` are the plain network. `rho` is validated non-negative and `rho=0` reduces to one ordinary pass.
- `use_revin=False` skips normalization as in the official model.
- The official network returns a flattened `[batch, channels * pred_len]` tensor and its trainer fits on pre-windowed arrays; TSFLab returns `[batch, pred_len, channels]` and uses the framework data pipeline, scaling, and loss (MSE) instead of the official dataset utilities.

## Citation

```bibtex
@inproceedings{ilbert2024samformer,
  title     = {{SAMformer}: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention},
  author    = {Romain Ilbert and Ambroise Odonnat and Vasilii Feofanov and Aladin Virmaux and Giuseppe Paolo and Themis Palpanas and Ievgen Redko},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024}
}
```

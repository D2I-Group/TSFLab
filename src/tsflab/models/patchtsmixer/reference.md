# PatchTSMixer — reference

## Differences in detail

Paper and the Hugging Face `transformers` implementation (pinned revision `3693f8d`, `models/patchtsmixer/modeling_patchtsmixer.py`, Apache-2.0) were checked; the local module is an independent rewrite of the supervised forecasting path. Patching (latest samples kept, `(seq_len - patch_len) // stride + 1` patches), the patch embedding, layer order (channel mixer in `mix_channel` mode, then patch mixer, then feature mixer), pre-LayerNorm residual blocks and the flatten-linear head follow the Hugging Face model. Recorded differences:

- Channel mixer gate order: Hugging Face `PatchTSMixerChannelFeatureMixerBlock` applies the gated attention before the MLP, whereas its patch and feature mixers apply it after. The local `AxisMixer` applies the gate after the MLP for all three axes (the local reading of paper Sec. 3.3.5), so `mix_channel` differs from Hugging Face in this one place.
- Normalization is the shared `revin` component without affine parameters, which matches the Hugging Face default `scaling="std"` (mean and biased standard deviation with `+1e-5`) applied per instance and channel; other scalers (`mean`, none), `observed_mask` handling and `norm_mlp="BatchNorm"` are not implemented (LayerNorm over `d_model`, `eps=1e-5`).
- Not implemented: the optional patch self-attention (`self_attn`), positional encoding, self-supervised masked pretraining, the `flatten` mode, the linear/classification/regression heads, distribution (Student-t) output, `prediction_channel_indices`, and `output_range`. Only MSE point forecasting is supported.
- Defaults differ from Hugging Face (`d_model=16` vs 8, `dropout` and `head_dropout` 0.1 vs 0.2, `patch_len=16` vs 8, stride 8) and Hugging Face's `init_std=0.02` initialization is not applied (PyTorch default initialization). The head applies dropout before flattening, which is element-wise equivalent to Hugging Face's dropout after flattening.
- The paper's online reconciliation head and hybrid channel modeling are not included.

## Citation

```bibtex
@inproceedings{ekambaram2023tsmixer,
  title     = {{TSMixer}: Lightweight {MLP}-Mixer Model for Multivariate Time Series Forecasting},
  author    = {Ekambaram, Vijay and Jati, Arindam and Nguyen, Nam and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2023}
}
```

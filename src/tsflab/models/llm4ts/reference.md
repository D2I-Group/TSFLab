# LLM4TS — reference

## Differences in detail

Paper (arXiv 2308.08469, Sec. 4, ACM TIST 2025) and the official code at revision `cb2426c` (`models/LLM4TS.py`, `layers/Embed.py`, `layers/RevIN.py`, `exp/exp_supervised_finetuning.py`, `exp/exp_long_term_forecasting.py`, `main.py`, `exp_settings_and_results/long_term_forecasting/ETTh1.json`; MIT license) were read; nothing was copied. The GPT-2 trunk is the shared local `gpt2_backbone` (checked once against Hugging Face `GPT2Model`, see that card).

- **Weights.** `spec.build` without the pinned artifact gives a random trunk for offline structure tests only.
- **PEFT.** Kaiming-uniform `A`, zero `B`, scale `alpha / r`. AdaLoRA (the official `main.py` default) is not implemented; the reported ETTh1 configuration uses LoRA.
- **Temporal encoding.** The official pipeline feeds continuous `timeF` marks to the learned calendar tables, so `.long()` truncates them to index 0 and the official temporal term is effectively one constant learned vector. Raw TSFLab marks are converted with `marks.adapt_tslib_marks` (year dropped); `freq="t"` adds a 15-minute-bin minute table. Marks absent means no temporal term.
- **Stages.** The official code runs alignment as a separate experiment and reloads its weights (minus the output layer) for each horizon, then switches from linear probing to fine-tuning at half of the forecasting epochs. Here `pretrain` (TSFLab `pretraining-stage`) runs `align_epochs` of alignment and `probe_epochs` of probing on the training loader with AdamW and MSE, without early stopping; the trainer's configured epochs, optimizer and loss perform fine-tuning. Defaults follow the official ETTh1 settings (alignment 5 epochs, lr 7.9e-5, wd 5.5e-4; probing 7 = 15 // 2 epochs, lr 1.8e-5, wd 1.5e-3).
- **Trainable sets.** Alignment trains encodings, LoRA, GPT-2 LayerNorms and positions and the alignment head; probing only the forecasting head; fine-tuning RevIN, encodings, LoRA and the forecasting head. The official fine-tuning leaves GPT-2 LayerNorms and positions frozen even though the paper describes LayerNorm tuning; this entry keeps the official behaviour. Fixed (sinusoidal) calendar tables never train.
- **Channels.** The official loader samples one channel per example (`return_single_feature`); here channels are flattened into the batch axis, the same computation with RevIN shared across channels (`RevIN(1)`). With `features = "MS"`, the probing stage still fits every channel.
- **Normalization.** Alignment uses statistics of the window plus the next `stride` values, as in the official code; RevIN uses `eps = 1e-5` with the official denormalization `(y - b) / (w + eps^2)`.

## Citation

```bibtex
@article{chang2025llm4ts,
  title   = {{LLM4TS}: Aligning Pre-Trained {LLMs} as Data-Efficient Time-Series Forecasters},
  author  = {Chang, Ching and Wang, Wei-Yao and Peng, Wen-Chih and Chen, Tien-Fu},
  journal = {ACM Transactions on Intelligent Systems and Technology},
  year    = {2025},
  doi     = {10.1145/3719207}
}
```

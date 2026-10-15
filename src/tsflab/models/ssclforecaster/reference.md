# SSCLForecaster — reference

## Differences in detail

- Sources read at revision `f394705adbf5217811276f5e59576c6e7ea7586c`: `models/AutoConCI.py`, `models/AutoConNet.py`, `layers/losses.py`, `layers/dilated_conv.py`, `layers/Embed.py`, `exp/exp_long_term_forecasting_with_AutoCon.py`, `run.py`, `scripts/AutoCon_mul_ETT.sh`; the paper's Sec. 3 (Eqs. 1-8, Fig. 4) and App. A.1-A.2 (Eqs. 9-13, Algorithm 1).
- The multivariate model is the channel-independent `AutoConCI`: one shared encoder and decoder, one autocorrelation per channel.
- Encoder input: `DataEmbedding` (circular token conv, sinusoidal position, linear `timeF` calendar features, dropout) of the normalized series, then the TS2Vec-style dilated convolution encoder (kernel 3, dilation `2**i`, widths `[d_model] * e_layers + [d_ff]`, GELU before each conv, 1x1 projection on the last block) and `Dropout(0.1)`.
- Decoder: GELU, `Linear(seq_len, pred_len)`, and per scale a moving average (kernel `scale + 1`), GELU, `Linear(d_ff, 1)` and the same moving average, summed (not averaged) over scales.
- Normalization options: the RevIN-style option uses the unbiased standard deviation plus `1e-5`; the decomposition option uses a kernel of 25 with the trend in the long branch.
- Autocorrelation: the statsmodels ACF of the training series after a moving average of length `seq_len + 1`, in absolute value.
- AutoCon: L2-normalized representations, temperature 1, the local term on `seq_len // 3` random steps per window and the global term on max-pooled windows per channel; each positive set is the highest-`r` partners (plus same-position partners) weighted by `r`, all non-self members in the denominator; loss `criterion + lambda * mean((local + global) / 2)`. The paper's Eq. (3) instead treats every lower-autocorrelation pair as a negative and averages over all `N - 1` partners.
- Defaults (`d_model = d_ff = 16`, `e_layers = 2`, `scales = [96]`, last-value normalization, `lambda = 1`) follow the ETTh1 multivariate script for horizon 96.
- Window positions: the official loader returns each window's start index; here the training setup orders training windows by first raw timestamp (one pass over the training loader, which also rebuilds the training series exactly from the windows) and the objective maps batch timestamps back to positions.
- The official loader drops incomplete batches; here a batch of one window simply has no global term.
- Calendar features are built from raw marks with `marks.adapt_tslib_marks`.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{park2024self,
  title     = {Self-Supervised Contrastive Learning for Long-term Forecasting},
  author    = {Park, Junwoo and Gwak, Daehoon and Choo, Jaegul and Choi, Edward},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

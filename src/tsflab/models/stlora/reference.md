# STLoRA — reference

## Paper

ST-LoRA: Low-rank Adaptation for Spatio-Temporal Forecasting, ECML-PKDD 2025, arXiv:2404.07919 (v1 2024-04, v2 2025-07).

## Differences in detail

- Sources read at revision `c89c21d1a9447ced37983fbd29cd09479c248669` (MIT): `src/model.py`, `src/models/lstm.py`, `src/utils/args.py`, `main.py`; paper Sec. 3.1-3.3 (Eqs. 2-5).
- Resolved from the official code: the NALL base weight is frozen while its bias trains; `A` uses Kaiming-uniform (`a = sqrt(5)`) and `B` zero initialisation with scaling `alpha / r` (`lora_alpha = 16`).
- The predictor starts with a 1x1 convolution over the feature axis at every step and node and ends with a NALL to one output feature; LeakyReLU slope 0.1 and dropout 0.3.
- LSTM backbone settings `init_dim = 32`, `hid_dim = 64`, `end_dim = 512`, two layers, dropout 0.3 come from `src/utils/args.py`; rank 16, four NALL layers, and one predictor block follow Sec. 4.2.
- The official `Node_Specific_Predictor` uses a single shared `A` (no node index), has no residual connections (LeakyReLU, dropout and BatchNorm after each layer), and the official `STLoRA` wrapper fuses as `Y_base + mean_k(Y_base * sigmoid(NSP_k(Y_base)))` with every block reading the backbone output.
- The official output reshape from `(B*N, ..., T)` to `(B, T, N, ...)` without a transpose mixes nodes and steps; here nodes and steps keep their axes.
- Eq. (2)'s complexity remark `O(r (d_in + d_out) + N r^2)` does not match its own definition `A_i in R^{r x d_in}`; the definition is implemented (`N r d_in` adapter parameters per layer).
- Eq. (4) concatenates `X` and `Z` per step along the feature axis, which needs `seq_len == pred_len` (the paper's 12-to-12 protocol, not stated); other lengths are rejected.
- Training uses the catalog loss (`mae` matches Eq. 5's data term). The official default (no `--frozen` / `--pre_train`) trains the backbone jointly from scratch, which is followed here.
- The paper also reports STGCN, GWNet, AGCRN, D2STGNN and STAEformer backbones; only LSTM is provided.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{ruan2024low,
  title   = {Low-rank Adaptation for Spatio-Temporal Forecasting},
  author  = {Ruan, Weilin and Chen, Wei and Dang, Xilin and Zhou, Jianxiang and Li, Weichuang and Liu, Xu and Liang, Yuxuan},
  journal = {arXiv preprint arXiv:2404.07919},
  year    = {2024}
}
```

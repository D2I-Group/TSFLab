# WaveRoRA — reference

## Paper

WaveRoRA: Wavelet Rotary Route Attention for Multivariate Time Series Forecasting (arXiv 2410.22649, 2024; revised 2024-11). Sources used: Sections III-IV (Eqs. 1-13, Alg. 1, Figs. 2-3) and Section V-C.

```bibtex
@article{liang2024waverora,
  title   = {WaveRoRA: Wavelet Rotary Route Attention for Multivariate Time Series Forecasting},
  author  = {Liang, Aobo and Sun, Yan and Guizani, Nadra},
  journal = {arXiv preprint arXiv:2410.22649},
  year    = {2024}
}
```

## Implementation mapping

Independent rewrite after reading the pinned official code (`Leopold2333/WaveRoRA` at `04b13249`, MPL-2.0): `models/WaveRoRA.py`, `layers/WaveRoRA_layer.py`, `layers/Attention.py` (`RouterAttention`), `layers/Transformer_EncDec.py` (`GatedAttentionLayer`, `Encoder`), `utils/rotation.py`, `configs/*.py`, `trainer/long_term_forecasting.py`, `scripts/LTSF/WaveRoRA/*.sh`. Nothing was copied or imported.

Resolved from the official code:

- Normalization subtracts the detached mean and divides by `sqrt(var + 1e-5)` (population variance).
- DWT: zero extension, coefficient lengths `floor((N + L - 1) / 2)`, order `[d_1, ..., d_J, a_J]`; the inverse drops the last sample of an approximation one longer than the next detail.
- Attention: `n_heads = 8`, scale `1 / sqrt(d_model / n_heads)` for both score maps, attention dropout on both maps after rotation, values equal to the input tokens, `Linear(d_model, d_model)` skip of the values, `SiLU(Linear(values))` gate; routing tokens are a learned `randn` parameter.
- Rotary positions: pairs of the router axis are rotated with angles from a reversed table, so variate `i` of `M` gets position `M - 1 - i` (base 10000).
- WaveNorm: `LN(x + drop(u))` then `LN(x + drop(conv2(drop(gelu(conv1(x))))))`, convolutions over the variate axis; no final encoder norm, no embedding dropout; calendar marks unused (`embed_type = 0`).
- Preset: `scripts/LTSF/WaveRoRA/ETTh.sh`, ETTh1, horizon 96 (`seq_len = 96`, Symlet-3, `wavelet_layers = 3`, `wavelet_dim = 64`, `d_model = 256`, `d_ff = 256`, `e_layers = 1`, `dropout = 0.3`, `ks = 1`, MSE loss, learning rate `1e-4`, batch 32).

## Differences in detail

- Alg. 1 projects values and per-head routing tokens and applies an output projection; the code (followed) does none of these.
- Section IV-D feeds a `(J + 1) D` token to RoRA; the code projects to `d_model` before and back after (`expand_projection`, `out_projection`), implemented.
- Eq. 13 is one linear map; the code predictor `Linear(D, 2D) -> GELU -> Linear(2D, H_j)` is implemented.
- Section V-C caps routing tokens at `min(10, ...)`; the code has no cap (14 for 321 variates). `router_num = 0` selects the code rule.
- Section V-C states a 4-level Symlet-3 DWT and `N = 2` layers for ETT; the scripts use 2-5 levels, Symlet-3 or Coiflet-3, and `e_layers = 1` for ETTh1 up to horizon 336.
- WaveNorm (Eq. 12) is undefined in the paper; the code's post-norm block with `ks = 3` in several scripts mixes neighbouring variates in file order. Implemented as `kernel_size`.
- Code bugs fixed: `--activation relu` inserts the class `nn.ReLU` and fails; odd `pred_len` gives `pred_len + 1` steps that break de-normalization (cropped here); one variate yields 0 routing tokens and breaks `RoPE1d` (minimum 2 here).
- Filters: standard tabulated Symlet-3, Coiflet-3, and Haar low-pass filters with high-pass and reconstruction filters from the quadrature-mirror relations; the official code reads them from PyWavelets through `pytorch_wavelets`. Filter buffers are kept in the state dict as `dwt.analysis`/`dwt.synthesis`.
- The cataloged `wavelet` and `haar_dwt1d` components were not reused: they pad odd lengths by replication or circularly rather than with zeros and lack Symlet-3/Coiflet-3.

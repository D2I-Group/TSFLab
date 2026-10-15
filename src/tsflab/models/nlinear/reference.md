# NLinear — reference

## Paper

- **Title**: Are Transformers Effective for Time Series Forecasting?
- **Venue**: AAAI 2023 (arXiv 2205.13504, 2022-05)
- **Abstract (shortened)**: Questions Transformer-based long-term forecasting (LTSF): permutation-invariant self-attention inevitably loses temporal order information even with positional encoding and sub-series tokens. The authors introduce LTSF-Linear, a set of embarrassingly simple one-layer linear models (including NLinear and DLinear), which outperform sophisticated Transformer LTSF models on nine real-life datasets, often by a large margin, and study how design elements affect temporal relation extraction.

## Implementation mapping

- Normalization: `center_on_last_value` takes the last lookback step per channel (detached), subtracts it from all steps, and returns it as the level.
- Temporal map: `ChannelWiseLinear(seq_len, pred_len, enc_in, individual)` applied along time on `[B, C, L]`; one shared `nn.Linear` by default, one per channel with `individual=True`.
- Restoration: `restore_last_value` adds the level back to every horizon step.
- Inputs are checked to have shape `[B, seq_len, enc_in]`; marks and decoder inputs are accepted and ignored.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZengCZ023,
  author    = {Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  title     = {Are Transformers Effective for Time Series Forecasting?},
  booktitle = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI} 2023},
  pages     = {11121--11128},
  publisher = {{AAAI} Press},
  year      = {2023},
  doi       = {10.1609/AAAI.V37I9.26317},
  url       = {https://doi.org/10.1609/aaai.v37i9.26317}
}
```

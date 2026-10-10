# TimeMosaic — reference

## Paper

TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and
Segment-wise Decoding (Ding et al., AAAI 2026, pp. 20790-20798; arXiv:2509.19406).

Fixed-length patching ignores the heterogeneity of local dynamics (losing detail in dense regions and
adding redundancy in stable ones) and the differing difficulty of short and long horizons. TimeMosaic
adapts patch granularity to local information density while preserving temporal continuity, and
decodes each horizon segment as a related subtask. It improves consistently on benchmarks, and a model
trained on a 321-billion-observation corpus is competitive with state-of-the-art foundation models.

## Implementation notes

- Regions: `seq_len // max(patch_sizes)`; each region yields `max / min` aligned tokens, so token count = regions * (max / min).
- Candidate embeddings for coarser sizes are repeated to match the finest token count before the hard (straight-through) selection.
- Segment `i` has length `pred_len // num_segments` plus one for the first `pred_len % num_segments` segments.
- `d_model` must be divisible by `num_heads`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/DingFHWWYZ26,
  author    = {Kuiye Ding and Fanda Fan and Chunyi Hou and Zheya Wang and Lei Wang and Zhengxin Yang and Jianfeng Zhan},
  title     = {TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding},
  booktitle = {Fortieth {AAAI} Conference on Artificial Intelligence, {AAAI} 2026},
  pages     = {20790--20798},
  publisher = {{AAAI} Press},
  year      = {2026},
  doi       = {10.1609/AAAI.V40I25.39218}
}
```

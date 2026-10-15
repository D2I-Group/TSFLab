# TexFilter — reference

## Paper

FilterNet: Harnessing Frequency Filters for Time Series Forecasting (Yi et al., NeurIPS 2024;
arXiv:2411.01623).

Transformer forecasters are vulnerable to high-frequency signals, costly, and under-use the full
spectrum. FilterNet introduces learnable frequency filters that selectively pass or attenuate signal
components: (i) a plain shaping filter with a universal frequency kernel, and (ii) a contextual shaping
filter whose filtered frequencies are examined for compatibility with the input for dependency
learning. The filters approximately surrogate linear and attention mappings, handle high-frequency
noise, and use the whole spectrum; eight benchmarks show strong effectiveness and efficiency.

## Implementation notes

- Embedding: complex weight `[seq_len // 2 + 1, embed_size // 2 + 1]`, initialized with scale `(seq_len // 2 + 1)^-0.5`.
- Context gains: two complex vectors over the output bins, initialized to `1 + 0j`; ReLU is applied separately to real and imaginary parts.
- Head: `Linear(embed_size, hidden_size) -> GELU -> Dropout -> Linear(hidden_size, pred_len)`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/0001FZHHL024,
  author    = {Kun Yi and Jingru Fei and Qi Zhang and Hui He and Shufeng Hao and Defu Lian and Wei Fan},
  title     = {FilterNet: Harnessing Frequency Filters for Time Series Forecasting},
  booktitle = {Advances in Neural Information Processing Systems 37, NeurIPS 2024},
  year      = {2024},
  url       = {http://papers.nips.cc/paper\_files/paper/2024/hash/6323d96f79d5d49e0d3fc88835c082cd-Abstract-Conference.html}
}
```

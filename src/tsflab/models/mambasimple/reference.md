# MambaSimple — reference

## Paper

Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv 2312.00752 (2023-12).

Subquadratic sequence models (linear attention, gated convolutions, structured SSMs) lag attention on discrete
modalities because they cannot do content-based reasoning. Mamba makes the SSM parameters functions of the input,
so the model selectively propagates or forgets information along the sequence, and uses a hardware-aware parallel
recurrent algorithm in a simplified architecture without attention or MLP blocks. It has linear scaling in sequence
length, about 5x Transformer inference throughput, and strong results on language, audio, and genomics.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2312-00752,
  author  = {Albert Gu and Tri Dao},
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  journal = {CoRR},
  volume  = {abs/2312.00752},
  year    = {2023},
  url     = {https://doi.org/10.48550/arXiv.2312.00752},
  doi     = {10.48550/ARXIV.2312.00752}
}
```

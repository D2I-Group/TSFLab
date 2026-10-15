# HDMixer — reference

## Paper

- **Title**: HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting
- **Venue**: AAAI 2024, pages 12608-12616
- **DOI**: 10.1609/aaai.v38i11.29155

## Abstract (shortened)

Length-fixed patches lose temporal boundary information such as complete peaks and periods, and existing patch methods focus on long-term dependencies across patches while neglecting short-term dependencies within patches and interactions among cross-variable patches. HDMixer is a pure-MLP model with a Length-Extendable Patcher (LEP) that enriches patch boundary information and a Hierarchical Dependency Explorer (HDE) that models within-patch, across-patch, and cross-variable dependencies. Experiments cover 9 real-world datasets.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/HuangSZCDZW24,
  author    = {Qihe Huang and Lei Shen and Ruixin Zhang and Jiahuan Cheng and
               Shouhong Ding and Zhengyang Zhou and Yang Wang},
  title     = {HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate
               Time Series Forecasting},
  booktitle = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI} 2024},
  pages     = {12608--12616},
  publisher = {{AAAI} Press},
  year      = {2024},
  doi       = {10.1609/AAAI.V38I11.29155}
}
```

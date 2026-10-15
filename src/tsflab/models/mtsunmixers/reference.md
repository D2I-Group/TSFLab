# MTSUNMixers — reference

## Paper

- **Title**: MTS-UNMixers: Multivariate Time Series Forecasting via Channel-Time Dual Unmixing
- **Venue**: arXiv preprint (2411.17770, 2024-11)
- **Sections used**: Sec. III-B (Eqs. 1-11), Sec. IV (Eqs. 12-21, Figs. 2-4), Sec. V.

## Equivalence evidence

Focused checks cover:

- the directional scan against a hand-written selective-SSM recurrence, and its causality;
- the Bi-Mamba gating and bidirectionality;
- the simplex constraints of `S_c`, `S_t`, and `S_p`;
- last-value patch padding;
- the dual-unmixing products and projections of Eqs. (17)-(20) with standardization;
- the L1 reconstruction objective of Eq. (21);
- channel mixing, gradients to every parameter, and the strict parameter schema.

## Citation

```bibtex
@article{zhu2024mtsunmixers,
  title   = {MTS-UNMixers: Multivariate Time Series Forecasting via Channel-Time Dual Unmixing},
  author  = {Zhu, Xuanbing and Shen, Dunbin and Rao, Zhongwen and Ma, Huiyi and Hao, Yingguang and Wang, Hongyu},
  journal = {arXiv preprint arXiv:2411.17770},
  year    = {2024}
}
```

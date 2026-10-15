# AirFormer — reference

## Paper

- **Title**: AirFormer: Predicting Nationwide Air Quality in China with Transformers
- **Venue**: AAAI 2023 (arXiv 2211.15979, 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.15979

## Abstract

AirFormer predicts nationwide air quality in China at fine spatial granularity covering thousands of locations.
It decouples learning into a bottom-up deterministic stage with two new self-attention mechanisms for efficient spatio-temporal representation, and a top-down stochastic stage with latent variables that captures the intrinsic uncertainty of air quality data.
On 4 years of data from 1,085 stations it reduces errors by 5-8% on 72-hour predictions versus the previous state of the art.

## Runtime contract

Inputs are `x_enc [B, seq_len, N]` and raw or node-structured covariates; the spatial preprocessing is `dartboard_mx [N, M, N]`. Output is `[B, pred_len, N]`.

## Citation

```bibtex
@inproceedings{liang2023airformer,
  author    = {Yuxuan Liang and Yutong Xia and Songyu Ke and Yiwei Wang and Qingsong Wen and Junbo Zhang and Yu Zheng and Roger Zimmermann},
  title     = {AirFormer: Predicting Nationwide Air Quality in China with Transformers},
  booktitle = {Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI 2023)},
  pages     = {14329--14337},
  year      = {2023},
  doi       = {10.1609/AAAI.V37I12.26676}
}
```

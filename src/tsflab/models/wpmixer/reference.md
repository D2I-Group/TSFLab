# WPMixer — reference

## Paper

WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting (AAAI 2025, arXiv 2412.17176).

MLP mixers are a promising alternative to Transformers for forecasting but have not reached their potential. WPMixer combines multi-resolution wavelet decomposition (time and frequency information), patching and embedding (extended look-back and local information), and MLP mixing (global information). It reports significant gains over MLP- and Transformer-based models for long-term forecasting at low computational cost.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/MuradAY25,
  author    = {Md Mahmuddun Nabi Murad and Mehmet Aktukmak and Yasin Yilmaz},
  editor    = {Toby Walsh and Julie Shah and Zico Kolter},
  title     = {WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series
               Forecasting},
  booktitle = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
               Conference on Innovative Applications of Artificial Intelligence,
               Fifteenth Symposium on Educational Advances in Artificial Intelligence,
               {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages     = {19581--19588},
  publisher = {{AAAI} Press},
  year      = {2025},
  url       = {https://doi.org/10.1609/aaai.v39i18.34156},
  doi       = {10.1609/AAAI.V39I18.34156}
}
```

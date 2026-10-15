# GOTSF — reference

## Paper

- **Title**: Goal-Oriented Time-Series Forecasting: Foundation Framework Design
- **Venue**: AAAI 2026 (arXiv 2504.17493, 2025-04)

## Abstract (shortened)

Conventional forecasting minimizes overall error without accounting for the varying importance of forecast ranges downstream.
The method partitions the prediction space into fine-grained segments during training, then reweights and aggregates them at inference to emphasize an application-specified target range, without retraining and without predefining the ranges.
Experiments on standard benchmarks and a new wireless communication dataset show gains inside regions of interest and in downstream task performance.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/FecheteSAPLDS26,
  author    = {Luca{-}Andrei Fechete and Mohamed Sana and Fadhel Ayed and Nicola Piovesan and
               Wenjie Li and Antonio De Domenico and Tareq Si Salem},
  title     = {Goal-Oriented Time-Series Forecasting: Foundation Framework Design},
  booktitle = {Fortieth {AAAI} Conference on Artificial Intelligence, {AAAI} 2026, Singapore, January 20-27, 2026},
  pages     = {21065--21073},
  publisher = {{AAAI} Press},
  year      = {2026},
  doi       = {10.1609/AAAI.V40I25.39249}
}
```

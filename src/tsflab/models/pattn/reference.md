# PAttn — reference

## Paper

- **Title**: Are Language Models Actually Useful for Time Series Forecasting?
- **Venue**: NeurIPS 2024 Spotlight (arXiv 2406.16964, 2024-06)
- **Abstract (shortened)**: Ablations of three popular LLM-based forecasters show that removing the LLM or replacing it with a basic attention layer does not degrade, and usually improves, accuracy. Pretrained LLMs do no better than models trained from scratch, do not represent sequential dependencies, and do not help in few-shot settings, at large computational cost. Simple patching-plus-attention encoders such as PAttn perform similarly to LLM-based forecasters.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/TanMGAH24,
  author    = {Mingtian Tan and Mike A. Merrill and Vinayak Gupta and Tim Althoff and Tom Hartvigsen},
  editor    = {Amir Globersons and Lester Mackey and Danielle Belgrave and Angela Fan and
               Ulrich Paquet and Jakub M. Tomczak and Cheng Zhang},
  title     = {Are Language Models Actually Useful for Time Series Forecasting?},
  booktitle = {Advances in Neural Information Processing Systems 37: Annual Conference
               on Neural Information Processing Systems 2024, NeurIPS 2024},
  year      = {2024},
  url       = {http://papers.nips.cc/paper\_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html}
}
```

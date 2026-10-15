# MAFS — reference

## Paper

Huang et al., "Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent
Cooperation", NeurIPS 2025.

Monolithic forecasters struggle to generalize across series with shifting patterns, statistics, and horizons.
MAFS decomposes forecasting into sub-tasks, each handled by an agent trained on a specific temporal
perspective; agents share information through a communication topology and a lightweight voting aggregator
combines their outputs. It outperforms single-model approaches on 11 benchmarks.

## Differences in detail

Pinned source inspected: `mafs_hetegenous_sub_task/models/Agent_iTrans_Cooperation.py`. The rewrite keeps
iTransformer-style variate-token agents, multi-scale specialization targets (Appendix A), layer-wise graph
communication (Eq. 4), masked symmetric normalized topology weights (Eq. 5), confidence blending, and an
input-conditioned global voter (Eqs. 6-7). The runner trains a single stage: the training objective sums the
configured criterion on the final forecast and the fixed-graph homogeneous prefix `specialization_loss`. This is
a joint multi-task approximation of the paper's separate ten-epoch specialization stage followed by
frozen-agent collaboration; there is no staged schedule or freezing.

## Citation

```bibtex
@inproceedings{huang2025mafs,
  author    = {Qihe Huang and Zhengyang Zhou and Yangze Li and Kuo Yang and Binwu Wang and Yang Wang},
  title     = {Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://papers.nips.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html}
}
```

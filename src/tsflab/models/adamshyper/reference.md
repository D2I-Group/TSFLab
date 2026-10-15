# AdaMSHyper — reference

## Citation

```bibtex
@inproceedings{shang2024adamshyper,
  title     = {{Ada-MSHyper}: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting},
  author    = {Shang, Zongjiang and Chen, Ling and Wu, Binqing and Cui, Dongliang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

## Differences in detail

Paper and official code (pinned revision `8efd34c`, `models/ASHyper.py`) were consulted for structure only; the official repository has no license file and nothing was copied. Where the official code departs from the paper, the local module follows the paper.

- Prediction head: the official forward sums three branches (a DLinear-style `seq_len -> pred_len` linear on the normalized input, a linear over the hypergraph-convolved node tokens initialized to the average, and a linear over hyperedge-attention tokens zero-padded to a hard-coded 80) and then applies another `pred_len -> pred_len` linear. The local model uses one linear over the concatenated node tokens of all scales and the attended hyperedge tokens, with no input skip.
- Incidence learning: the official adjacency is made binary with `torch.where`, which blocks gradients to the node and hyperedge embeddings, scales logits by a fixed `alpha=3`, and drops empty hyperedge columns. The local incidence uses `softmax(relu(E_node E_hyper^T))` without the factor, adds a straight-through term so the soft scores receive gradients, and masks empty hyperedges out of attention, constraints and the output.
- Constraint losses: the official node loss is `|mean(v_i - e_j)|` over member pairs; its hyperedge loss sums over all ordered pairs (including a hyperedge with itself) with a hard-coded margin 4.2 and `|mean|` per pair, combined with unit weights; validation weights `0.2 * MSE + 0.8 * |constraint|`; the training loop keeps a second optimizer for the constraint term. The local node loss is the mean absolute feature gap of a node to its hyperedges and the hyperedge loss averages over non-empty distinct pairs (Eq. 10-12), combined as `const_weight * (lambda_balance * node + (1 - lambda_balance) * hyperedge)` (Eq. 13) and exposed as `aux_loss` during training only, added to the MSE criterion by the shared trainer.
- Multi-scale construction: local `ScaleConv` is a strided Conv1d + BatchNorm + ELU per scale with window = stride and no final LayerNorm; the official default CSCM is `Bottleneck_Construct` (Linear down/up projection around the convolutions, three convolution layers in `Conv_Construct`, final LayerNorm). Default windows are `[4, 4]` (three scales), `hyper_num = [50, 20, 10]`; `eta` corresponds to the official `k = 3`.
- Widths: the hypergraph feature width is the channel count (`enc_in`) rather than a `d_model`-wide projection; `d_embed` is the node and hyperedge embedding width (official `d_model`). The official per-scale `linhy`/`linnod` layers are unused in its forward and omitted. The official hard-coded sizes (80, 76, 100, 320) are replaced by sizes derived from `seq_len` and `hyper_num`.
- Normalization is the shared `revin` component without affine parameters, equal to the official mean/std normalization (`+1e-5`).

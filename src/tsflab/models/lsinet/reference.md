# LSINet — reference

## Paper

Zhang et al., AAAI 2025 (arXiv 2602.01585). Linear models rival Transformers for long-term forecasting but
interact over time only implicitly through stacked MLPs. LSINet adds a Multihead Sparse Interaction Mechanism
(MSIM) that learns important time-step connections via a sparsity-induced Bernoulli distribution, kept sparse
by a self-adaptive regularization loss, and Shared Interaction Learning (SIL) that shares interactions for
efficiency and convergence. It is an MLP-only model reported to beat linear and Transformer models in accuracy and efficiency.

## Differences in detail

- **No auxiliary training loss.** ASRL (Eq. 7, a BCE term between the router's soft edge probabilities and an intermittently refreshed top-K target) is a training-time regularizer; the fixed four-input `forward` contract returns only the forecast, so it is omitted. The router still enforces sparsity at inference through its top-k hard mask.
- **Single logit.** The paper (Eqs. 3-4) forms a two-way softmax `{c^0_ij, c^1_ij}` per patch pair; here each pair has one logit read through `sigmoid`/binary-Gumbel noise, the equivalent binary-concrete reparameterization. Tensor shapes differ from the official `[..., 2]` logits.
- **Memory-to-head reshape.** The official `.view(n_heads, num_patches, -1)` reinterprets a `(num_patches, n_heads * d_model)` buffer, mixing patch and head axes. The paper text and Fig. 3(b) describe one `d_model` embedding per patch per head; the local router gives each head its own slice of each patch's memory embedding.
- **No ablation branches.** The official `Self_Attention_Mechanism` (SAM) swap and the `resdual_block` extra-MSIM-residual ablation are training-script flags outside Fig. 3 (SAM appears only in Table 3; `resdual_block` defaults off). LSINet here is the Fig. 3a architecture with `e_layers` stacked STI modules; the paper uses one (`e_layers=1`) on all reported datasets.
- **Heads.** The official `Flatten_Head` also supports `individual` and variable-group-decomposition (`var_decomp`) heads; `flatten_forecast_head` covers shared and `individual` and omits `var_decomp`.
- **Shared component.** `sparse_connection_router` was extracted from MSIM as a paper-neutral, input-independent sparse adjacency over N positions (position-memory table -> pairwise relational MLP -> top-k/Gumbel discretization); LSINet is currently its only consumer.
- Marks and decoder arguments are accepted and ignored, as in the paper and official code.

## Citation

```bibtex
@inproceedings{zhang2025lightweight,
  title     = {A Lightweight Sparse Interaction Network for Time Series Forecasting},
  author    = {Zhang, Xu and Wang, Qitong and Wang, Peng and Wang, Wei},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39}, number = {12}, pages = {13304--13312},
  year      = {2025}
}
```

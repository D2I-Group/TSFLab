# adaptive_node_embedding_adjacency — reference

## Origin and granularity

Five spatiotemporal models (`origin_models`) carried a verbatim copy of this three-operation block;
commit `dd8d8196` ("refactor(components): gated_dilated_conv and
adaptive_node_embedding_adjacency") extracted it from `gwnet`, `dfdgcn`, `himnet`,
`d2stgnn` and `agcrn`. The dual-embedding form is the Graph WaveNet self-adaptive
adjacency; the single-embedding form is the node-adaptive graph used by AGCRN.
The commit message and the module docstring record the models but not a per-copy
paper citation, so the paper attributions above are the conventional ones and the
repository only verifies the model list. The cut stops at the adjacency: how the
embeddings are parameterised and initialised (e.g. `himnet` passes a batched
per-sample "meta" embedding) and how the adjacency is mixed with static or dynamic
supports stay model-local; the identity/Chebyshev basis and node-adaptive filter
stacked on it in `agcrn` and `himnet` were later extracted as
`node_adaptive_graph_conv`, which now calls this function for both models. `mtgnn` has a different graph constructor (frozen reference `_RefGraphConstructor`) and was
deliberately left out of the extraction. `adamshyper` and `stdmae` consume it too.

## Invariants and equivalence evidence

- A contract check written at extraction verified the `[6, 6]` shape and dtype,
  row sums of 1, non-negativity, the batched `[2, 6, 4]` self-form giving
  `[2, 6, 6]`, that the dual form equals `softmax(relu(e @ t), -1)` with `t` used
  as supplied, that the single form equals the dual form with `e.T`, and seeded
  values pinned as a regression value; a gradient check verified non-zero
  gradients to both embeddings. There was no error-path check (the function does
  not validate).
- Equivalence against frozen verbatim pre-extraction copies of `gwnet`, `dfdgcn`,
  `himnet`, `d2stgnn` and `agcrn` was checked at extraction: identical state-dict
  keys and values, identical eval outputs (`atol=1e-6`) and identical gradients
  for each model. That frozen-copy test passed in the full suite run of
  2026-10-03 before the test suite was consolidated. `adamshyper` and `stdmae`
  adopted the component later and have no frozen pre-extraction copy.

## Variants and options

- Dual form (`target` given): independent source/target tables, asymmetric
  adjacency (`gwnet`, `dfdgcn`, `d2stgnn`, `stdmae`, `adamshyper`; the last gives a
  rectangular `[nodes, hyper_nodes]` result).
- Single form (`target=None`): symmetric scores before the softmax, supports
  batched embeddings (`agcrn`, `himnet`, through `node_adaptive_graph_conv`).
- There is no temperature, top-k sparsification, or normalization variant; those
  would be new components.

## Related components

`diffusion_conv` (consumes such adjacencies as supports),
`graph_utils` (static
supports mixed with this learned one), `regularized_adaptive_graph_conv` (linear-time
node-embedding graph), `gated_dilated_conv` (co-extracted in the same commit),
`sparse_connection_router` (learned sparse adjacency over positions),
`node_adaptive_graph_conv` (node-adaptive Chebyshev filter built on the single form).
- `adj_norm`: normalizes the support this component produces.

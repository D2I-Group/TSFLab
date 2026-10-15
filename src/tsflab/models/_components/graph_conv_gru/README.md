---
name: "graph_conv_gru"
description: "Graph-convolutional GRU gating: r, u = sigmoid(G_g([x, h])), c = tanh(G_c([x, r * h])), h' = u * h + (1 - u) * c around caller-supplied graph filters. Use for recurrent encoders or decoders on node graphs (DCRNN, AGCRN family); not for cells with other gate equations or non-graph sequences."
---

# graph_conv_gru

## What it does

The recurrence shared by graph-convolutional GRUs. A gate filter `G_g` (output
width `2H`) and a candidate filter `G_c` (output width `H`) replace the affine
maps of a GRU; for `x [B, N, I]` and `h [B, N, H]`:

```
r, u = split_half( sigmoid(G_g(concat[x, h])) )      # first half reset, second half update
c    = tanh( G_c(concat[x, r * h]) )
h'   = u * h + (1 - u) * c
```

Only this gating is shared. The filters (diffusion, Chebyshev, node-adaptive,
dynamic-graph, polynomial-feature + linear, ...) and their graph arguments are
supplied by the caller, so the component composes with any graph operator that
maps `[B, N, I + H]` to `[B, N, out]`.

## When to use

Use for a recurrent spatiotemporal encoder or decoder whose GRU gates should
mix information over a graph, with the filter chosen independently. Do not use
when the paper's cell differs from the equations above (another gate order,
reset applied after the filter, extra gates or normalisation), or for
non-graph sequences (use a standard GRU).

## Interface

`graph_gru_step(x, state, gates, candidate) -> Tensor`

- `x` `[B, N, I]`, `state` `[B, N, H]`; `gates` and `candidate` are callables on
  the concatenated `[B, N, I + H]` tensor returning `[B, N, 2H]` and `[B, N, H]`.
  The gate output is halved with `chunk(2, dim=-1)`: first half reset `r`, second
  half update `u`. Returns the next state `[B, N, H]`. Stateless, no parameters,
  no validation (shape errors surface from torch).

`GraphConvGRUCell(gates, candidate)` (an `nn.Module`)

- Registers the two filter modules as `gates` and `candidate`, so their
  parameters appear in the state dict as `gates.*` then `candidate.*`.
- `forward(x, state, *graph)` calls `graph_gru_step` with
  `gates(joined, *graph)` and `candidate(joined, *graph)`; `graph` is any number
  of extra positional arguments (none for filters holding their supports as
  buffers, an embedding table, a support tensor, a tuple or a list of supports).
- Subclass it to keep a model's class name and constructor: build the two
  filters, call `super().__init__(gates, candidate)`, then set any width
  attribute the model reads (`hidden`, `hidden_dim`, `units`).

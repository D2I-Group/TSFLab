# graph_utils — reference

## Origin and granularity

The file began as an `_external/graph_utils.py` module in commit `3f6b3120` ("Port 16
non-duplicate CauAir models"), a vendored helper. It was then rebuilt as an independent
composition over `adj_norm` and `graph_spectral` (later history commits `3ac9e264`, `fba5fa99`, and
`17835348` "refactor(graph): reuse shared adjacency supports", which switched `d2stgnn`,
`dcrnn`, `dfdgcn`, `gwnet`, and `stgcn` to it; `stgcn` now uses `graph_spectral` directly). The module docstring says it is
"an independent composition of the repository's dense adjacency normalizers and
spectral helpers"; the original external source is not recorded in history. The boundary is
the list-of-supports interface (mode names follow the DCRNN/Graph WaveNet convention).
The actual matrix math lives in `adj_norm` and `graph_spectral`; what stays model-local is
how supports are used (stacking, adaptive supports, graph convolution layers).
Current consumers: `gwnet`, `dcrnn`, `dfdgcn`, `d2stgnn`, `st_ssdl`, `stdmae`.

## Invariants and equivalence evidence

- Contract check (at extraction), for all seven modes: the support count,
  `[N, N]` float32 finite shape, COO equal to dense, and `adj_to_supports` dtype;
  `symadj == I - normlap`, `origin` has unit diagonal, the reverse walk equals
  `transition_matrix(A.T)`, `cheb_poly` equals `chebyshev_polynomials`, the four
  `ValueError` cases (unknown mode, unknown `return_type`, non-square, non-finite), and seeded
  `fwd`, `rev`, `scalap` outputs pinned as regression values.
- Finiteness check: `adj_to_supports` returns float32
  and equals `transition_matrix(A)` and `transition_matrix(A.T)` on a graph with an
  isolated node; `scalap` is finite on an identity graph; `cheb_poly(.., 0)` and an
  unknown `adj_type` raise.
- Equivalence of the extracted `gwnet`, `dfdgcn`, `d2stgnn` (outputs and gradients)
  against frozen verbatim pre-extraction copies that call `adj_to_supports` was
  checked at extraction; those frozen-copy tests, like the checks above, passed in
  the full suite run of 2026-10-03 before the test suite was consolidated.

## Variants and options

- `doubletransition` (default of `adj_to_supports`) for diffusion models (`gwnet`,
  `dcrnn`, `d2stgnn`, `dfdgcn`); `symadj` is used by `st_ssdl`; `scalap` for Chebyshev use.
- `return_type="coo"` for sparse consumers; dense is the default.
- For direct Chebyshev stacks as tensors use `graph_spectral.chebyshev_supports` instead.

## Related components

`graph_spectral` (the Laplacian scaling and Chebyshev math this module wraps; use it
directly for tensor Chebyshev stacks, as `stgcn` does), `adj_norm` (the row-sum normalizers
behind the other modes), `diffusion_conv` (consumes `doubletransition` supports),
`adaptive_node_embedding_adjacency` (learned instead of precomputed adjacency).

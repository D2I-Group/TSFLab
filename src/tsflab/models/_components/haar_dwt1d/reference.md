# haar_dwt1d — reference

## Origin and granularity

Extracted from `swift` (introduced in the automated intake commit `6b491e13`),
where the forward transform splits the (reversible-normalized) window into
sub-series, a small convolution fuses them, per-sub-series linear maps predict the
output sub-series, and the inverse reconstructs the horizon. Only the Haar pair is
here; the fusion convolution, the linear maps, and the `_half_length`
bookkeeping stay in `swift`. For multi-level or longer filters use `wavelet`.

## Invariants and equivalence evidence

- Lossless reconstruction was checked for lengths 8 and 9 on `[2, 3, T]` tensors,
  and `[2, 4, 6, 8]` against the closed form.
- Contract check (at extraction): shapes for lengths 8 and 9, round trip, energy preservation (even length),
  gradient flow, empty state dicts, and the two `ValueError` cases (`T = 1`,
  mismatched shapes); seeded outputs, including an odd-length input, were
  pinned as regression values. These checks passed in the full suite run of 2026-10-03 before the test suite was consolidated.

## Variants and options

None. Single level, Haar only, last axis only, no boundary modes. For
multi-level, db2/db4, or undecimated transforms see `wavelet`. Its
`DecimatedWaveletTransform("haar")` uses the same filters and the same
last-sample replication for odd lengths, so a single level should agree with
this pair numerically (by inspection of the code; no test compares them), but it
returns a list `[approx, detail]`, is a stateful module that remembers the
padding for `reconstruct`, and does not need a `length` argument.

## Related components

`wavelet` (multi-level, db2/db4, a-trous; overlaps at level 1 as described
above), `series_decomposition` (moving-average trend/seasonal split, not
decimated), `revin` (applied before the DWT in `swift`).

# SCINet alignment

Reference: https://github.com/cure-lab/SCINet at
`02e6b0af2d58243de09aaa1eac3840237b659847`, `models/SCINet.py`.
This replaces the earlier Time-Series-Library provenance link. No external source
is copied into the production package.

Matched operations: num_levels-1 recursive tree depth; replication padding with
kernel_size+1 total padded samples; kernel_size then width-three convolutions;
d = odd * exp(phi(even)), c = even * exp(psi(odd)); even = c + U(d), odd = d - P(c);
interleaving; tree/input residual; unbiased-free temporal projection; optional
original sine-then-cosine PE and trainable-affine RIN; full-history two-stack concat.

Numerical audit uses matched active weights at seq_len168/pred_len24 for default,
PE+RIN and two-stack paths. Output tolerance accounts for kernel-one Conv1d versus
Linear arithmetic. The original RIN modifies a tensor needed for input gradients
in place; TSFLab uses out-of-place operations. The RIN case checks output parity
and finite local input gradients; other cases compare input gradients numerically.

The two-stack reference returns final and intermediate predictions. The ordinary
TSFLab forecast API returns only the final forecast; intermediate-output training
is outside this adapter. Preset num_stacks=1 has no such supervision difference.

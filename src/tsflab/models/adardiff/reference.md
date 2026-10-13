# Equation map and declared recipe

Source: https://arxiv.org/html/2608.28134v1, Sections3.1,3.3 and Appendices B,C.

- Difference: z_t=x_t-sum_j delta_j*x_(t-j), j=1..P. Left boundary repeats the first observation.
- Residual backbone: one shared Linear(L-1,H), no channel mixing.
- Initial-condition term: c_k=sum_(j>=k)delta_j*x_(L+k-j).
- Kernel: h_0=1; h_n=sum_(j<=min(n,P))delta_j*h_(n-j).
- Reconstruct: causal linear convolution of h and predicted residual+c.
- Phase1: MSE against target differences, reconstruction bypassed.
- Phase2: ordinary MSE against forecast targets, no target consumed by inference.

Polynomial Newton doubling computes the finite inverse of1-sum_j delta_j*z^j,
using sufficiently zero-padded linear convolutions. A frequency reciprocal on a
finite FFT grid is a circular inverse and is not substituted. Four float64 kernel
and gradient cases, including a stable .95 lag1 counterexample, and full forecast
recurrence equivalence were checked; this is not a numerical stability guarantee
for all coefficients. Large unconstrained unstable roots may overflow and fail.

The paper uses30/10maximum epoch phases with Adam and early stopping. It selects
period, initialization, RevIN, l1 reparameterization, LR and batch on validation.
This entry exposes a fixed declared subset, not the complete paper grid. A targeted
run with lr.001,batch32,period24/168 and seed42 is an adaptation. Native ETTh1
uses first14400rows rather than the17420rows listed in the paper dataset table.
Checkpoint choice for phase1 uses residual validation patience3; optimizer state
is not transferred between phases. These differences limit claims of exact parity.

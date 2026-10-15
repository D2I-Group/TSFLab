# GWNetGMM — reference

## Differences in detail

- Implementation: independent rewrite of Sections 3.1-3.3 (Eqs. 3-10) of arXiv 2604.16084v2. The official repository (`Weijiang-Xiong/OpenSkyTraffic`, revision `824534a112b37973481eedaa705d1a14279f0743`) has no license file, so it is recorded as `NOASSERTION` and used as reference only: `skytraffic/models/layers/gmmpred.py`, `skytraffic/models/gwnet.py`, `skytraffic/models/gwnet_gmm.py` and `config/gwnet/GWNET_GMM.py` were read to resolve omissions; nothing was copied or imported.
- Naming: the catalog entry is named after the official `GWNET_GMM` class and config, the configuration linked from the repository README for this paper. The method is backbone-agnostic (the paper also adapts STID, STAEformer, MTGNN, an LSTM-GCN and HiMSNet).
- Resolved from the official code: the projection is followed by LayerNorm, ReLU and dropout before the three branches; `spacing_scale` is initialized to zero; the GWNet backbone uses only the adaptive adjacency (`aptonly=True`, random node embeddings of width 10, diffusion order 2), `in_dim=2` (value and time of day), 32 residual/dilation channels, 256 skip and 512 end channels, 4 blocks of 2 layers for 12 input steps, a one-step left pad of the input, and dropout 0.3 in both the graph convolution and the head; the point estimate is the mixture mean (`mcd_estimation=False`).
- Head initialization: zero weights for offsets and log-variances, mixing bias `1/K`, so mixing weights are equal at start.
- Shared components, checked against the reference: `diffusion_conv.DiffusionConv2d` matches its graph convolution term order, 1x1 projection and dropout; `adaptive_node_embedding_adjacency` matches `softmax(relu(E1 @ E2), dim=1)`; `marks.to_spatiotemporal` supplies `[value, time_in_day, day_in_week]` and the first `in_dim` channels are used. `gated_dilated_conv` was not reused because it left-pads every layer (constant length), while the reference uses valid convolutions whose BatchNorm statistics and skip alignment depend on the shrinking time axis.
- Loss: the official `GMMPredictionHead.losses` (`skytraffic/models/layers/gmmpred.py`) drops the `log(2 pi)` constant of Eq. 5; it is kept here. Missing labels are masked only when they are NaN (the official pipeline also turns its dataset null value into NaN). Validation uses the configured loss.
- Graph: the predefined road-graph supports and the `randomadj=False` SVD initialization are not exposed (the headline configuration disables them); the runner's `adj_mx` is accepted but unused.
- Depth: `layers` is an explicit parameter and the model rejects windows whose `seq_len + 1` exceeds the receptive field instead of computing the layer count; shorter windows are left-padded to the receptive field.
- Evaluation: the catalog output type is `point`; the runner evaluates the mixture mean with point metrics. The mixture is available from `Model.mixture` (normalized space), but CRPS, HDPS intervals, mAW and mCCE (Sec. 3.3-3.4) are not wired into TSFLab evaluation.

## Verification

Checked properties: the receptive field and shapes; the prior initialization; Eqs. 6-8 and 10 in the head; the NLL against the explicit mixture density with NaN masking; the prior NLL; adaptive-graph node mixing; the declared training objective; gradients. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@article{xiong2026unveiling,
  title   = {Unveiling Stochasticity: Universal Multi-modal Probabilistic Modeling for Traffic Forecasting},
  author  = {Xiong, Weijiang and Fonod, Robert and Geroliminis, Nikolas},
  journal = {arXiv preprint arXiv:2604.16084},
  year    = {2026}
}
```

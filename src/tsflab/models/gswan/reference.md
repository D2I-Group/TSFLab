# GSWaN — reference

## Differences in detail

- Implementation: independent rewrite of Sections 3.2.2-3.2.6 (Eqs. 4-8, Figs. 4-5) of arXiv 2302.09956. The official repository (`aprbw/G-SWaN`, revision `d95c2af5349187373c7e484073bcc0472fe71597`) has no license file, so it is recorded as `NOASSERTION` and used as reference only: `GSWaN_models.py`, `model.py`, `util.py`, `GSWaN_loader.py` and `run_pems08.txt` were read to resolve omissions; nothing was copied or imported.
- Resolved from the official code (`run_pems08.txt`, used for every dataset): 40 hidden channels (also the node-embedding width, since embeddings are added to the latent input), 320 skip and 640 end channels, 4 blocks of 2 layers with kernel 2 for 12 input steps (input left-padded to the 13-step receptive field), 3 attention heads plus one plain diffusion head per support, diffusion order 2, softmax temperature 7, dropout 0.3 on the attention weights and the SGT output, concatenation (`CAT`) head aggregation, no attentive head pooling, Mish activations, BatchNorm, random node-embedding initialization (`--randomadj`), doubletransition road supports; keys and queries have `2 x hidden` channels and a kernel spanning the remaining time length; the skip path reads only the last time step of every layer; the last layer has no SGT.
- Shared components, checked against the reference: `graph_utils.adj_to_supports(..., "doubletransition")` matches the official `[asym_adj(A), asym_adj(A^T)]`; `adaptive_node_embedding_adjacency` matches `softmax(relu(e1 @ e2), dim=1)`; `marks.to_spatiotemporal` supplies the value and calendar channels. `gated_dilated_conv` and `diffusion_conv` were not reused: the former pads every layer while G-SWaN uses valid convolutions whose lengths fix the key/query kernel widths, and the latter has no attention heads or batched supports.
- Graph nodes come from `num_nodes` injected by the runner (falling back to `enc_in`).
- Time feature: the paper calls the second input "time of day", but the official loader feeds the elapsed fraction of the week (`(index mod 2016) / 2015`, assuming the series starts at a week boundary). `time_of_week` here is `(weekday + time_in_day) / 7` from calendar marks (denominator 7 days instead of `2016 - 1` steps). When a dataset supplies node-structured covariates, their first channel is used as the time input.
- Training: the three official augmentations (spatial occlusion, temporal permutation, uniform noise), training-mean missing-value filling, learning-rate decay 0.97 and gradient clipping 3 belong to the official training loop and are not part of this model; the paper trains with MAE.
- The official loader clips validation and test forecasts to `[0, max]` of the whole series (test included); this entry outputs unclipped forecasts.

## Verification

Checked properties: the valid dilated lengths against the official key/query kernel widths; the support list and term count; Eq. 6 masking and source normalization; zero attention heads reduce the SGT to diffusion over the supports; node embeddings enter keys and queries (Eqs. 7-8); the time covariates; output shape and spatial mixing; gradients to node embeddings and attention heads; runner-injected adjacency. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@inproceedings{prabowo2023gswan,
  title     = {Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting},
  author    = {Prabowo, Arian and Shao, Wei and Xue, Hao and Koniusz, Piotr and Salim, Flora D.},
  booktitle = {Proceedings of the 8th ACM/IEEE Conference on Internet of Things Design and Implementation},
  pages     = {93--104},
  doi       = {10.1145/3576842.3582362},
  year      = {2023}
}
```

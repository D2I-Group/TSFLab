# Extralonger — reference

## Implementation mapping

- **Global-local spatial attention (`GLSAtt`)**:
  `(Softmax(alpha_local) + Softmax(alpha_global)) @ V / 2` matches the official
  `AttentionLayer` with `mode="spatial"` exactly and is the extracted
  `graph_masked_attention` component (`GlobalLocalGraphAttention`).
- **Fusion**: the official `forward`'s hand-set weights (temporal 0.25, spatial
  0.25, mixed 0.50).
- **Temporal route**: `t_input` (`nn.Linear(num_nodes, input_embedding_dim)`),
  concatenated time-of-day/day-of-week embeddings, self-attention blocks over
  time, `t_output` (`model_dim -> num_nodes`) and, when `seq_len != pred_len`,
  `Linear(seq_len, pred_len)`; matches `t_input`/`attn_layers_t`/`t_output`.
- **Spatial route**: `s_input` (`nn.Linear(seq_len, model_dim -
  spatial_embedding_dim)`), node-identity embedding, global-local attention
  blocks over nodes, `s_output` to `pred_len`; matches
  `s_input`/`attn_layers_s`/`s_output`.
- **Mixed route**: pre-attention temporal features through a second temporal
  stack (`mix_temporal_layers`), transposed so features are tokens of width
  `seq_len`, a second stack (`mix_spatial_layers`, fixed 2 heads as hard-coded
  officially), and `mix_proj` (`Linear(model_dim, num_nodes)` then
  `Linear(seq_len, pred_len)`); matches
  `attn_layers_mix_t`/`attn_layers_mix_s`/`mix_proj`.

## Differences in detail

- `input_noise` is an `nn.Parameter` broadcast onto the raw input before the
  routes split, trained like a bias; at initialization it is a fixed additive
  offset. Removing it does not change the route/fusion architecture.
- `num_nodes`/`adj_mx` are runner-injected, as for STID/STAEformer/VisiFold.
- Preset widths: `input_embedding_dim=8`, `tod_embedding_dim=4`,
  `dow_embedding_dim=4`, `spatial_embedding_dim=8`, `feed_forward_dim=16`,
  `num_heads=2`, `num_layers=1`, for CPU contract checks; the official
  PEMS04/PEMS08/SEATTLE presets (`model/config.yaml`) use widths up to 128-256
  and `feed_forward_dim` up to 1024, with horizons set by `train.py`'s `-i`/`-o`.

## Citation

Zhang, Z., E, S., Meng, F., Zhou, J., Han, W. "Extralonger: Toward a Unified Perspective of Spatial-Temporal Factors for Extra-Long-Term Traffic Forecasting." NeurIPS 2024 Workshop. arXiv:2411.00844.

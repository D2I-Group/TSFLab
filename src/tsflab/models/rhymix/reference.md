# Equation and implementation map

Source: [RhyMix v1](https://arxiv.org/html/2607.08234v1), Section 3.1.2.
This independent implementation repairs earlier local reproductions; it is not
established as an exact implementation of the authors' unpublished code.

|Operation|Source symbol|Contract|
|---|---|---|
|RevIN and inverse|`Model.forward`|Population standard deviation plus epsilon; per-channel affine transform and inverse|
|Channel mixing|`channel_mlp`|Residual C-to-r-to-C bottleneck, dropout, scale initially .05|
|Cyclic context|`cyclic`, `period_gate`|Four hourly periods, implicit fixed phase0, per-channel weights and past/future tables|
|Contextual path|`TMSBlock`|Two RMS-normalized depthwise-dilation1/2/4/8 residual blocks with adaptive channel gates|
|Four heads|`ForecastHeads.forward`|Mixed input or temporal output; retain local/periodic residuals and channel-resolved future prior|
|Output gates|`path_gate`, `hybrid_gate`|Normalized statistics averaged across channels, softmax4 and softmax2|

CNN geometry is inferred as two channel-by-time planes with a 3x3 kernel.
The four-head module is shared between paths following the paper's shared-head
description; parameter tying is not established by author code. Channel-attention
hidden width uses the same bounded bottleneck rule as the mixer. The paper does
not specify every such dimension. Model-local reflection padding is retained
instead of substituting an edge-padded moving-average component. Normalization
uses the written population-std-plus-epsilon equations rather than the existing
RevIN component's variance-plus-epsilon convention.

The paper recipe specifies Huber delta1, AdamW lr.0007/weight_decay.0001,
50 maximum epochs, patience6, cosine minimum1e-6 and gradient clipping1.
Batch32 and seed values are declared experimental choices, not author settings.
Use split-independent input/target windows for the paper protocol; train-only
scaling is unchanged. Never tune these choices using test outcomes.

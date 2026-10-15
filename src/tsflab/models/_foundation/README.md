# Foundation runtime boundary

This directory is infrastructure, not a model family or a second catalog. Every
TSFLab model and method remains a flat entry in `src/tsflab/models/<slug>/`.

`FoundationModel` adapts one official pretrained runtime to the canonical
`[batch, time, channels]` forecasting contract. Official packages load only from
an explicit local checkpoint path; they may use their standard cache, but this
layer never downloads, copies, converts, or republishes weights. Provider imports
are optional and lazy.

## Official runtime adapters

| Runtime | Official package | TSFLab boundary |
| --- | --- | --- |
| Chronos | `chronos-forecasting` | `ChronosRuntime` wraps `BaseChronosPipeline` |
| TimesFM | `timesfm[torch]` | `TimesFMRuntime` wraps TimesFM 2.5 |
| Moirai | `uni2ts` | `MoiraiRuntime` accepts an official Moirai 2 forecast object |

Moirai should run in an environment compatible with the selected Uni2TS release;
its current PyTorch constraint conflicts with TSFLab's main environment. This
boundary keeps that incompatibility explicit instead of silently changing the
repository runtime.

## Related flat catalog entries

These local implementations study foundation-model architectures but do not load
the released pretrained checkpoints: [TiRex](../tirex/README.md),
[Kronos](../kronos/README.md), [SEMPO](../sempo/README.md),
[SymTime](../symtime/README.md), and [TimeCAP](../timecap/README.md).

These are methods that operate above or alongside a foundation forecaster:
[CoRA](../cora/README.md), [TSRAG](../tsrag/README.md), and
[OccamVTS](../occamvts/README.md). They remain ordinary model entries because
their algorithms and verification contracts are independently useful. A future
official-backbone experiment may compose them with this runtime boundary without
moving or duplicating their implementations.

## Integration rule

An official pretrained model becomes a normal flat model entry only when its
model card, local checkpoint facts, runtime factory, inference-only capability,
and verification evidence are complete. The shared interface does not register
providers by itself.

"""Torch-free table of executable slot options and their compatibility rules.

Slots follow the model-card composition order. Each option names the cataloged
components it wraps, the card-documented interface it relies on, and (for
options that cannot run) the reason. ``check_assignment`` returns every reason a
slot assignment cannot be built, so a spec fails validation before any run.

Uniform per-slot interface (see ``adapters.py``), with ``x`` ``[B, L, C]``:

- normalization: ``norm(x) -> x'`` and ``denorm(y [B, H, C]) -> y``; ``rescale(scale)``.
- decomposition: ``split(x') -> [part, ...]`` (each ``[B, L, C]``).
- temporal: ``encode(part [B, L, C]) -> z [B, C, D, P]`` (one module per part).
- channel: how weights relate across channels (independent / individual / mixing).
- head: ``project(z [B, C, D, P_total]) -> forecast`` with the head's output kind.
- loss: a ``training.loss`` name (kept in the run config, not in the model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

SLOTS = ("normalization", "decomposition", "temporal", "channel", "head", "loss")


#: Components imported by the adapters, per option (card-documented interfaces).
@dataclass(frozen=True)
class SlotOption:
    slot: str
    name: str
    components: tuple[str, ...] = ()
    interface: str = ""
    aliases: tuple[str, ...] = ()
    unsupported: str = ""
    output: str = "point"  # head options only: point | quantile | distribution
    extra: dict[str, str] = field(default_factory=dict)


def _o(*args, **kwargs) -> SlotOption:
    return SlotOption(*args, **kwargs)


OPTIONS: dict[str, dict[str, SlotOption]] = {
    "normalization": {
        o.name: o
        for o in (
            _o("normalization", "none", (), "identity"),
            _o(
                "normalization",
                "revin",
                ("revin",),
                "RevIN(C, affine=False): norm/denorm on [B, T, C]; denorm of [B, H, Q, ...] per slice",
                aliases=("component:revin",),
            ),
            _o(
                "normalization",
                "last_value_center",
                ("last_value_center",),
                "center_on_last_value(x)->(x, level [B,1,C]); restore_last_value(y, level)",
                aliases=("component:last_value_center",),
            ),
        )
    },
    "decomposition": {
        o.name: o
        for o in (
            _o("decomposition", "none", (), "single branch"),
            _o(
                "decomposition",
                "series_decomposition",
                ("series_decomposition",),
                "SeriesDecomposition(kernel)(x [B,L,C]) -> (residual, trend); two branches",
                aliases=("component:series_decomposition",),
            ),
            _o(
                "decomposition",
                "wavelet",
                ("wavelet",),
                "",
                aliases=("component:wavelet",),
                unsupported="the wavelet component works on [B, C, L] and returns subbands whose length changes per level "
                "(decimated) or that have no inverse (undecimated); no uniform [B, L, C] branch interface exists",
            ),
        )
    },
    "temporal": {
        o.name: o
        for o in (
            _o(
                "temporal",
                "linear",
                (),
                "no encoder: z = history as [B, C, 1, L]; the head's linear map is the temporal model",
                aliases=("local:linear",),
            ),
            _o(
                "temporal",
                "channel_wise_linear",
                ("channel_wise_linear",),
                "ChannelWiseLinear(L, hidden, C, individual)([B, C, L]) -> z [B, C, 1, hidden]",
                aliases=("component:channel_wise_linear",),
            ),
            _o(
                "temporal",
                "tst_transformer",
                ("tst_transformer", "positional_encoding"),
                "patches [B*C, P, patch_len] -> Linear -> +positional_encoding -> TSTEncoder([B*C, P, d_model]) -> z [B, C, d_model, P]",
                aliases=("component:tst_transformer", "component:patchtst", "patchtst"),
            ),
            _o(
                "temporal",
                "mamba",
                ("mamba",),
                "patches [B*C, P, patch_len] -> Linear -> MambaResidualBlock x n_layers ([B*C, P, d_model]) -> z [B, C, d_model, P]",
                aliases=("component:mamba",),
            ),
            _o(
                "temporal",
                "gated_dilated_conv",
                ("gated_dilated_conv",),
                "Conv1d lift [B*C, 1, L] -> hidden; gated_dilated_conv(x, filter_conv, gate_conv) with dilations 1,2,4..; z [B, C, hidden, L]",
                aliases=("component:gated_dilated_conv",),
            ),
            _o(
                "temporal",
                "mixer_block",
                ("mixer_block",),
                "MixerBlock(L, C, hidden, dropout) x n_layers ([B, L, C]) -> z [B, C, 1, L]; mixes channels",
                aliases=("component:mixer_block",),
            ),
        )
    },
    "channel": {
        o.name: o
        for o in (
            _o(
                "channel",
                "independent",
                (),
                "channel-independent: weights shared across channels",
                aliases=(
                    "local:independent",
                    "local:channel-independent-shared-weights",
                    "component:channel_wise_linear",
                ),
            ),
            _o(
                "channel",
                "individual",
                (),
                "per-channel head weights (individual=True on FlattenForecastHead / ChannelWiseLinear)",
                aliases=("local:individual",),
            ),
            _o(
                "channel",
                "mixing",
                ("mixer_block",),
                "cross-channel mixing, only through mixer_block's feature MLP",
                aliases=("local:mixing", "component:mixer_block"),
            ),
        )
    },
    "head": {
        o.name: o
        for o in (
            _o(
                "head",
                "flatten_forecast_head",
                ("flatten_forecast_head",),
                "FlattenForecastHead(individual, C, nf, H)(z [B,C,D,P]) -> [B,C,H] -> [B,H,C]",
                aliases=("component:flatten_forecast_head",),
            ),
            _o(
                "head",
                "quantile_head",
                ("flatten_forecast_head", "quantile_head"),
                "FlattenForecastHead point base [B,H,C] -> unsqueeze(-1) -> QuantileHead -> [B,H,C,Q]",
                aliases=("component:quantile_head",),
                output="quantile",
            ),
            _o(
                "head",
                "gaussian_parameter_head",
                ("gaussian_parameter_head",),
                "GaussianParameterHead(D*P, H)(z flattened [B,C,D*P]) -> (loc, scale) [B,C,H] -> [B,H,C,2]",
                aliases=("component:gaussian_parameter_head",),
                output="distribution",
            ),
        )
    },
}

DEFAULTS = {
    "normalization": "none",
    "decomposition": "none",
    "channel": "independent",
    "head": "flatten_forecast_head",
    "loss": "mse",
}

#: Training loss required by each head output kind (None: any point loss).
OUTPUT_LOSS = {"point": None, "quantile": "quantile", "distribution": "nll_gaussian"}

_PROBABILISTIC_LOSSES = {"quantile", "nll_gaussian"}


def canonical(slot: str, option: str) -> str:
    """Resolve a spec option (``component:revin``, ``none``, alias) to its option name."""
    text = str(option).strip()
    if text in OPTIONS.get(slot, {}):
        return text
    for opt in OPTIONS.get(slot, {}).values():
        if text in opt.aliases:
            return opt.name
    raise ValueError(
        f"slot {slot}: {option!r} is not an executable option; choose from {sorted(OPTIONS.get(slot, {}))}"
    )


def option(slot: str, name: str) -> SlotOption:
    return OPTIONS[slot][name]


def assignment_components(assignment: dict[str, str]) -> tuple[str, ...]:
    """Catalog components a (canonical) assignment imports, sorted."""
    used: set[str] = set()
    for slot in ("normalization", "decomposition", "temporal", "head"):
        used.update(OPTIONS[slot][assignment[slot]].components)
    if assignment["channel"] == "mixing":
        used.update(OPTIONS["channel"]["mixing"].components)
    return tuple(sorted(used))


def all_components() -> tuple[str, ...]:
    return tuple(
        sorted({c for opts in OPTIONS.values() for o in opts.values() if not o.unsupported for c in o.components})
    )


def check_assignment(assignment: dict[str, str]) -> list[str]:
    """Return every reason the canonical slot assignment cannot be built (empty = ok)."""
    problems = _option_problems(assignment)
    if problems:
        return problems
    temporal, channel, head = assignment["temporal"], assignment["channel"], assignment["head"]
    problems.extend(_pairing_problems(temporal, channel, head))
    problems.extend(_loss_problems(assignment, head))
    return problems


def _option_problems(assignment: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for slot in SLOTS:
        if slot == "loss":
            continue
        opt = OPTIONS[slot].get(assignment.get(slot, ""))
        if opt is None:
            problems.append(f"slot {slot}: unknown option {assignment.get(slot)!r}")
        elif opt.unsupported:
            problems.append(f"slot {slot}: {opt.name} is not executable: {opt.unsupported}")
    return problems


def _pairing_problems(temporal: str, channel: str, head: str) -> list[str]:
    problems: list[str] = []
    if temporal == "mixer_block" and channel != "mixing":
        problems.append(
            "temporal mixer_block mixes channels in its feature MLP; set channel to mixing (it cannot be channel-independent)"
        )
    if channel == "mixing" and temporal != "mixer_block":
        problems.append(
            f"channel mixing is only provided by mixer_block; temporal {temporal} is channel-independent (use channel independent or individual)"
        )
    if head == "gaussian_parameter_head" and channel == "individual":
        problems.append(
            "GaussianParameterHead has shared weights only; per-channel (individual) weights are unsupported with it"
        )
    return problems


def _loss_problems(assignment: dict[str, str], head: str) -> list[str]:
    problems: list[str] = []
    loss = assignment.get("loss", DEFAULTS["loss"])
    required = OUTPUT_LOSS[OPTIONS["head"][head].output]
    if required is not None and loss != required:
        problems.append(
            f"head {head} emits {OPTIONS['head'][head].output} output and needs loss {required!r}, got {loss!r}"
        )
    if required is None and loss in _PROBABILISTIC_LOSSES:
        problems.append(f"loss {loss!r} needs a quantile_head or gaussian_parameter_head, got {head}")
    return problems


def resolve_assignment(slots: dict[str, list[str] | str]) -> tuple[dict[str, str], list[str]]:
    """Canonicalize a spec ``[slots]`` table into one option per slot plus errors.

    Absent slots take the defaults; ``temporal`` is required. Each slot takes one
    option for an executable composition (lists with several options are sweeps
    and are rejected here with a precise message).
    """
    errors: list[str] = []
    resolved: dict[str, str] = {}
    for slot in SLOTS:
        value, error = _resolve_slot(slot, slots.get(slot))
        if error is None:
            resolved[slot] = value
        else:
            errors.append(error)
    for slot in slots:
        if slot not in SLOTS:
            errors.append(f"unknown slot {slot!r}; choose from {list(SLOTS)}")
    return resolved, errors


def _resolve_slot(slot: str, raw: list[str] | str | None) -> tuple[str | None, str | None]:
    """``(option, None)`` for one spec slot, or ``(None, error)``."""
    if raw is None or raw == []:
        if slot == "temporal":
            return None, "slot temporal: required for an executable composition"
        return DEFAULTS[slot], None
    return _resolve_given_slot(slot, raw)


def _resolve_given_slot(slot: str, raw: list[str] | str) -> tuple[str | None, str | None]:
    items = [raw] if isinstance(raw, str) else list(raw)
    if len(items) != 1:
        return None, f"slot {slot}: an executable composition takes exactly one option, got {items}"
    text = str(items[0])
    try:
        if slot == "loss":
            return text.removeprefix("loss:"), None
        return canonical(slot, text), None
    except ValueError as exc:
        return None, str(exc)


#: Card prefixes for temporal options that are not ``component:<name>``.
_CARD_TEMPORAL_PARTS = {"linear": "local:linear", "tst_transformer": "component:tst_transformer"}


def card_composition(assignment: dict[str, str]) -> list[str]:
    """Six-slot ``composition`` list for a model card from a canonical assignment."""

    def part(slot: str) -> str:
        name = assignment[slot]
        if slot == "loss":
            return f"loss={'loss:' + name}"
        if name == "none":
            return f"{slot}=none"
        opt = OPTIONS[slot][name]
        if slot == "channel":
            return f"{slot}=local:{name}"
        if slot == "temporal" and name in _CARD_TEMPORAL_PARTS:
            return f"{slot}={_CARD_TEMPORAL_PARTS[name]}"
        del opt
        return f"{slot}=component:{name}"

    return [part(slot) for slot in SLOTS]

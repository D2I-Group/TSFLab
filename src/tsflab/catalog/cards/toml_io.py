"""Minimal, deterministic TOML writer for card files (reading uses ``tomllib``).

Cards hold only strings, integers, floats, booleans, lists of those, nested
tables, and arrays of tables, so a small emitter keeps the dependency surface
empty and the output stable (keys keep their insertion order).
"""

from __future__ import annotations

import json
import math
from typing import Any


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("cards cannot store NaN or infinity")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return _container(value)


def _container(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    if isinstance(value, dict):  # inline table
        return "{ " + ", ".join(f"{_key(k)} = {_scalar(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"unsupported TOML value {value!r}")


def _key(key: str) -> str:
    return key if key.replace("_", "").replace("-", "").isalnum() else json.dumps(key)


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


def _emit(lines: list[str], table: dict[str, Any], prefix: str, inline: tuple[str, ...]) -> None:
    _emit_plain(lines, table)
    for key, value in table.items():
        if isinstance(value, dict) and value:
            _emit_table(lines, key, value, prefix, inline)
    for key, value in table.items():
        if _is_table_array(value):
            _emit_table_array(lines, key, value, prefix, inline)


def _emit_plain(lines: list[str], table: dict[str, Any]) -> None:
    plain = {k: v for k, v in table.items() if v is not None and not isinstance(v, dict) and not _is_table_array(v)}
    for key, value in plain.items():
        lines.append(f"{_key(key)} = {_scalar(value)}")


def _emit_table(lines: list[str], key: str, value: dict[str, Any], prefix: str, inline: tuple[str, ...]) -> None:
    lines.append("")
    lines.append(f"[{prefix}{_key(key)}]")
    if f"{prefix}{key}" in inline:  # one inline table per entry
        for name, entry in value.items():
            lines.append(f"{_key(name)} = {_scalar(entry)}")
    else:
        _emit(lines, value, f"{prefix}{key}.", inline)


def _emit_table_array(
    lines: list[str], key: str, value: list[dict[str, Any]], prefix: str, inline: tuple[str, ...]
) -> None:
    for item in value:
        lines.append("")
        lines.append(f"[[{prefix}{_key(key)}]]")
        _emit(lines, item, f"{prefix}{key}.", inline)


def dumps(data: dict[str, Any], *, inline: tuple[str, ...] = ()) -> str:
    """Serialize ``data``; for tables named in ``inline`` each entry is one inline table."""
    lines: list[str] = []
    _emit(lines, data, "", inline)
    return "\n".join(lines).strip() + "\n"

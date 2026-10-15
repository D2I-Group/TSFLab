"""Resolve explicitly selected, importable execution plugins for any queue mode."""

import inspect
from importlib import import_module


def load_executor(reference):
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("executor must be an importable module:callable reference")
    module, name = reference.split(":")
    if not module or not name or not all(part.isidentifier() for part in (*module.split("."), *name.split("."))):
        raise ValueError("invalid executor reference")
    value = import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise ValueError("executor reference is not callable")
    inspect.signature(value).bind("directory", cancelled=lambda: False)
    return value

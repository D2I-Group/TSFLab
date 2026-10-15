"""Shared argument spellings for public commands.

Every command spells the same idea the same way: a single dataset or model is a
positional name, the forecast length is ``--horizon``, an output path is ``--out``,
the run-records root is ``--records``, and structured output is ``--json``.
Retired spellings stay accepted as hidden aliases so old scripts keep working.
"""

from __future__ import annotations

import argparse

HIDDEN = argparse.SUPPRESS


def add_option(parser, flag: str, dest: str, *aliases: str, **kwargs):
    """Add ``flag`` (stored in ``dest``); ``aliases`` parse the same way but stay out of help.

    The aliases are registered on the one action, so ``required``, ``append`` and types
    behave identically; only the help and error messages name the current spelling.
    """
    if kwargs.get("action") is None:
        kwargs.setdefault("metavar", flag.lstrip("-").upper())
    action = parser.add_argument(flag, *aliases, dest=dest, **kwargs)
    action.option_strings = [flag]
    return action


def add_name(parser, dest: str, *, help: str, flag: str, metavar: str | None = None) -> None:
    """A positional name that also accepts its former option spelling (``flag``) hidden."""
    parser.add_argument(dest, nargs="?", metavar=metavar or dest, help=help)
    parser.add_argument(flag, dest=f"{dest}_option", help=HIDDEN, default=None)


def resolve_name(parser, args, dest: str, *, required: bool = True, default=None) -> None:
    """Merge the positional and its hidden option spelling into ``args.<dest>``."""
    value = getattr(args, dest) or getattr(args, f"{dest}_option", None) or default
    if required and not value:
        parser.error(f"the {dest} is required: give it as the positional <{dest}>")
    setattr(args, dest, value)

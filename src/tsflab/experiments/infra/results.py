"""Shared result/error envelope for API calls and CLI adapters."""

from dataclasses import asdict, dataclass
from typing import Any


class ContractError(ValueError):
    """A pluggable service returned data outside its declared contract."""


@dataclass(frozen=True)
class OperationResult:
    ok: bool
    data: Any = None
    error: dict | None = None
    exit_code: int = 0
    diagnostics: str = ""
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)


def validate_execution_result(value):
    from collections.abc import Mapping

    if not isinstance(value, Mapping) or type(value.get("ok")) is not bool:
        raise ContractError("executor result must be a mapping with boolean 'ok'")
    return dict(value)


def error_info(error, *, fallback="operation failed"):
    if isinstance(error, dict):
        return {
            "type": str(error.get("type", "OperationFailed")),
            "message": str(error.get("message", fallback)),
        }
    if isinstance(error, BaseException):
        return {"type": type(error).__name__, "message": str(error)}
    return {
        "type": "OperationFailed",
        "message": str(error) if error is not None else fallback,
    }


def invoke(function, *args, **kwargs):
    """Opt into one envelope without changing legacy domain APIs or spawning CLI."""
    try:
        data = function(*args, **kwargs)
        ok = data.get("ok", True) if isinstance(data, dict) else True
        if type(ok) is not bool:
            raise ContractError("result 'ok' must be boolean")
        error = data.get("error") if isinstance(data, dict) and not ok else None
        if not ok and error is None:
            error = {
                "type": "OperationFailed",
                "message": "operation reported failure; inspect data",
            }
        return OperationResult(
            ok=ok,
            data=data,
            error=error_info(error) if not ok else None,
            exit_code=0 if ok else 1,
        )
    except Exception as exc:
        return OperationResult(
            ok=False,
            error={"type": type(exc).__name__, "message": str(exc)},
            exit_code=2,
        )

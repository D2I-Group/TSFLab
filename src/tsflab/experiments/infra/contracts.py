"""Small structural contracts for composing services without a shared base class."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol


class Cancellation(Protocol):
    def __call__(self) -> bool: ...


class Executor(Protocol):
    def __call__(self, directory: str, *, cancelled: Cancellation) -> Mapping[str, Any]: ...


class MetricsSink(Protocol):
    def start(self, step: int = 0) -> None: ...
    def log(self, values: Mapping[str, float], step: int) -> None: ...
    def close(self, failed: bool = False) -> None: ...


class RuntimeState(Protocol):
    def runtime_state_dict(self) -> Any: ...
    def load_runtime_state_dict(self, state: Any) -> None: ...


class FileCancellation:
    """Explicit, persistent cancellation signal; querying it has no side effects."""

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self) -> bool:
        return self.path.exists()

    def request(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()


def any_cancelled(*signals: Callable[[], bool]) -> Cancellation:
    """Compose caller, queue, or application signals without changing globals."""
    return lambda: any(signal() for signal in signals)

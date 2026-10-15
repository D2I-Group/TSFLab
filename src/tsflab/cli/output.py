"""Context-local structured output for CLI adapters; no subprocess round-trip."""

from contextvars import ContextVar

CAPTURE_CHILD_OUTPUT = ContextVar("capture_child_output", default=False)
STRUCTURED_OUTPUT = ContextVar("structured_output", default=None)


def publish(value):
    STRUCTURED_OUTPUT.set(value)
    return value

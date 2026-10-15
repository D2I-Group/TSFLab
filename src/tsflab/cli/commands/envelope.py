"""Versioned CLI result using the same envelope as direct Python invocation."""

import io
import json
import threading
from contextlib import redirect_stderr, redirect_stdout

from tsflab.cli.output import CAPTURE_CHILD_OUTPUT, STRUCTURED_OUTPUT
from tsflab.experiments.infra.results import OperationResult, error_info

_LOCK = threading.RLock()


def _run_captured(main, args, output, diagnostics):
    """Run ``main(args)`` with captured output; return its exit code, error, and structured data."""
    error = None
    # CLI output capture is serialized; concurrent integrations use invoke(API).
    with _LOCK:
        capture = CAPTURE_CHILD_OUTPUT.set(True)
        structured = STRUCTURED_OUTPUT.set(None)
        try:
            with redirect_stdout(output), redirect_stderr(diagnostics):
                try:
                    code = main(args) or 0
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 2
                except Exception as exc:
                    code = 2
                    error = {"type": type(exc).__name__, "message": str(exc)}
            data = STRUCTURED_OUTPUT.get()
        finally:
            STRUCTURED_OUTPUT.reset(structured)
            CAPTURE_CHILD_OUTPUT.reset(capture)
    return code, error, data


def _textual_data(output):
    # Compatibility for older commands whose domain output is still textual.
    try:
        return json.loads(output.getvalue())
    except ValueError:
        return {"text": output.getvalue().strip()}


def _command_error(code, error, data, output, diagnostics):
    error = error or (data.get("error") if isinstance(data, dict) else None)
    return error or {
        "type": "CommandError",
        "message": diagnostics.getvalue().strip() or output.getvalue().strip() or f"exit {code}",
    }


def envelope(args):
    from tsflab.cli.main import main

    output, diagnostics = io.StringIO(), io.StringIO()
    code, error, data = _run_captured(main, args, output, diagnostics)
    if data is None:
        data = _textual_data(output)
    if code:
        error = _command_error(code, error, data, output, diagnostics)
    result = OperationResult(
        ok=code == 0,
        data=data,
        error=error_info(error) if code else None,
        exit_code=code,
        diagnostics=diagnostics.getvalue().strip(),
    )
    print(json.dumps({**result.to_dict(), "command": args}, ensure_ascii=False))
    return code

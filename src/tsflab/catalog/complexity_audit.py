"""Static audit of code complexity in the method layer.

Two limits keep the codebase from growing more complex as methods are added:

* every function in ``src/tsflab/models`` has a cyclomatic complexity of at most
  ``MAX_FUNCTION_COMPLEXITY`` (one plus the branch points, as in McCabe's measure);
* every method package (``src/tsflab/models/<method>/``) has at most
  ``MAX_METHOD_LINES`` lines of code (blank lines, comments, and docstrings are not
  counted, so the limit does not depend on formatting).

The limits sit just above the code after the 2026-10 refactor (functions up to 9,
method packages up to 440 lines of code), so the existing methods pass and a change that adds
a more complex function or a larger method is asked to split it, to reuse shared
components, or to move parameter checks into the params schema. Shared components
(``_components``) and other private
packages follow the function limit only.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

from tsflab.core.paths import repository_root

MAX_FUNCTION_COMPLEXITY = 10
MAX_METHOD_LINES = 450

_BRANCHES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.IfExp,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.comprehension,
    ast.match_case,
)


def cyclomatic_complexity(node: ast.AST) -> int:
    """McCabe complexity of one function: 1 plus its branch points, not counting nested functions."""
    total = 1
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(child, _BRANCHES):
            total += 1
            if isinstance(child, ast.comprehension):
                total += len(child.ifs)
        elif isinstance(child, ast.BoolOp):
            total += len(child.values) - 1
        stack.extend(ast.iter_child_nodes(child))
    return total


def function_complexities(path: Path) -> list[tuple[str, int, int]]:
    """(qualified name, line, complexity) of every function and method in a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    out: list[tuple[str, int, int]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                out.append((name, child.lineno, cyclomatic_complexity(child)))
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")

    visit(tree, "")
    return out


_NON_CODE = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
_SCOPES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _SCOPES) and ast.get_docstring(node, clean=False) is not None:
            first = node.body[0]
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def code_lines(source: str) -> int:
    """Number of lines holding code; blank lines, comments, and docstrings are not counted."""
    lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _NON_CODE:
            lines.update(range(token.start[0], token.end[0] + 1))
    return len(lines - _docstring_lines(ast.parse(source)))


def audit_complexity(root: Path | None = None) -> list[str]:
    root = root or repository_root()
    models = root / "src" / "tsflab" / "models"
    errors: list[str] = []
    for path in sorted(models.rglob("*.py")):
        rel = path.relative_to(root)
        for name, line, cc in function_complexities(path):
            if cc > MAX_FUNCTION_COMPLEXITY:
                errors.append(
                    f"{rel}:{line} {name} has cyclomatic complexity {cc} "
                    f"(limit {MAX_FUNCTION_COMPLEXITY}); split it into smaller functions"
                )
    for package in sorted(p for p in models.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        lines = sum(code_lines(f.read_text(encoding="utf-8", errors="ignore")) for f in package.rglob("*.py"))
        if lines > MAX_METHOD_LINES:
            errors.append(
                f"{package.relative_to(root)} has {lines} lines of code (limit {MAX_METHOD_LINES}); "
                "move reusable parts to shared components"
            )
    return errors


def main() -> int:
    errors = audit_complexity()
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"Complexity OK: functions <= {MAX_FUNCTION_COMPLEXITY}, method packages <= {MAX_METHOD_LINES} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

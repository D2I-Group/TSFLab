"""Static complexity audit of the method layer (tsflab.catalog.complexity_audit)."""

import ast
from pathlib import Path

from tsflab.catalog import complexity_audit as ca


def _cc(src: str) -> int:
    return ca.cyclomatic_complexity(ast.parse(src).body[0])


def test_cyclomatic_complexity_counts_branch_points():
    assert _cc("def f():\n    return 1\n") == 1
    assert _cc("def f(x):\n    if x and x > 1:\n        return 1\n    for i in x:\n        pass\n") == 4
    # nested functions are measured on their own
    assert _cc("def f():\n    def g(x):\n        if x:\n            pass\n    return g\n") == 1


def test_audit_flags_complex_functions_and_large_methods(tmp_path: Path, monkeypatch):
    pkg = tmp_path / "src" / "tsflab" / "models" / "toy"
    pkg.mkdir(parents=True)
    branches = "\n".join(f"    if x == {i}:\n        return {i}" for i in range(ca.MAX_FUNCTION_COMPLEXITY + 1))
    (pkg / "model.py").write_text(f"def f(x):\n{branches}\n")
    errors = ca.audit_complexity(tmp_path)
    assert any("cyclomatic complexity" in e for e in errors)
    (pkg / "model.py").write_text("x = 1\n" * (ca.MAX_METHOD_LINES + 1))
    assert any("lines of code" in e for e in ca.audit_complexity(tmp_path))
    padded = '"""Doc."""\n# comment\n\n' + "x = 1\n" * ca.MAX_METHOD_LINES
    (pkg / "model.py").write_text(padded)
    assert ca.audit_complexity(tmp_path) == []


def test_code_lines_skip_blank_lines_comments_and_docstrings():
    source = '"""Module."""\n\n# note\ndef f(\n    a,\n    b,\n):\n    """Doc."""\n    return a + b\n'
    assert ca.code_lines(source) == 5


def test_repository_passes_complexity_audit():
    assert ca.audit_complexity() == []


def test_functions_nested_in_blocks_are_measured(tmp_path: Path):
    path = tmp_path / "nested.py"
    branches = "\n".join(f"            if x == {i}:\n                return {i}" for i in range(3))
    path.write_text(f"def outer():\n    with open('f'):\n        def inner(x):\n{branches}\n")
    assert [name for name, _, _ in ca.function_complexities(path)] == ["outer", "outer.inner"]


def test_audit_covers_framework_modules(tmp_path: Path):
    pkg = tmp_path / "src" / "tsflab" / "cli"
    pkg.mkdir(parents=True)
    (tmp_path / "src" / "tsflab" / "models").mkdir()
    branches = "\n".join(f"    if x == {i}:\n        return {i}" for i in range(ca.MAX_FUNCTION_COMPLEXITY + 1))
    (pkg / "command.py").write_text(f"def f(x):\n{branches}\n")
    assert any("src/tsflab/cli/command.py" in e for e in ca.audit_complexity(tmp_path))

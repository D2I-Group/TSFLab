"""Resolve repository resources in a checkout or an installed distribution."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def repository_root() -> Path:
    """Return the canonical checkout root or the wheel's read-only asset root."""
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "configs").is_dir() and (checkout / "src" / "tsflab" / "models").is_dir():
        return checkout
    packaged = Path(str(files("tsflab.assets")))
    if not (packaged / ".packaged-assets").is_file():
        raise RuntimeError("TSFLab repository resources are unavailable")
    return packaged


def is_packaged_root(root: Path | None = None) -> bool:
    """Return whether ``root`` is the immutable asset set bundled in a wheel."""
    candidate = repository_root() if root is None else root
    return (candidate / ".packaged-assets").is_file()


def require_checkout(operation: str) -> Path:
    """Reject repository-mutating operations from an installed wheel."""
    root = repository_root()
    if is_packaged_root(root):
        raise RuntimeError(f"{operation} modifies the catalog and requires a TSFLab git checkout")
    return root


def working_root() -> Path:
    """Return a writable execution root without writing into installed assets."""
    root = repository_root()
    return Path.cwd().resolve() if is_packaged_root(root) else root

"""Scientific reproducibility identities, independent of record persistence."""

import hashlib
from functools import lru_cache
from pathlib import Path

from tsflab.experiments.infra.storage import canonical_hash


def dataset_fingerprint(config) -> dict:
    """Hash explicit local dataset bytes; synthetic datasets are config-defined."""
    path = Path(config.dataset.path).expanduser() if config.dataset.path else None
    files = []
    if path is not None and path.exists():
        sources = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for source in sources:
            metadata = source.stat()
            digest = _file_digest(str(source.resolve()), metadata.st_size, metadata.st_mtime_ns)
            files.append(
                {
                    "path": str(source.relative_to(path) if path.is_dir() else source.name),
                    "sha256": digest,
                }
            )
    return {
        "sha256": canonical_hash({"dataset": config.dataset.model_dump(mode="json"), "files": files}),
        "files": len(files),
    }


def code_fingerprint() -> str:
    """Include uncommitted source changes and the locked dependency environment."""
    from tsflab.core.paths import repository_root

    root = repository_root()
    digest = hashlib.sha256()
    from tsflab.core.paths import is_packaged_root

    if is_packaged_root():
        code_root = Path(__file__).resolve().parents[2]
        paths = sorted(
            p
            for name in (
                "catalog",
                "cli",
                "core",
                "data",
                "experiments",
                "models",
                "realtime",
                "release",
                "research",
                "agent",
            )
            for p in (code_root / name).rglob("*.py")
        )
    else:
        code_root = root
        paths = sorted((root / "src").rglob("*.py"))
    paths += [root / name for name in ("uv.lock", "pyproject.toml") if (root / name).exists()]
    for path in paths:
        digest.update(str(path.relative_to(code_root) if path.is_relative_to(code_root) else path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_fingerprint() -> dict:
    """Record installed scientific/runtime versions, independently of a lockfile."""
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "pydantic"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


@lru_cache(maxsize=4096)
def _file_digest(path: str, size: int, modified_ns: int) -> str:
    """Reuse content hashes within a process only while file metadata is unchanged."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

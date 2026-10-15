"""Explicit, checksum-verified artifact storage for model weights and tokenizers."""

from __future__ import annotations

from pathlib import Path

from tsflab.catalog.registry.models import ModelArtifact, ModelSpec
from tsflab.release.hub.fetch import default_cache_root, download, sha256_file


def artifact_path(spec: ModelSpec, artifact: ModelArtifact, cache_root: Path) -> Path:
    """Return a traversal-safe cache path for one pinned artifact."""
    revision = "".join(c if c.isalnum() or c in "._-" else "_" for c in artifact.revision)
    return cache_root / "models" / spec.module.rsplit(".", 1)[-1] / revision / artifact.filename


def artifact_status(spec: ModelSpec, cache_root: Path | None = None) -> list[dict[str, object]]:
    """Describe declared artifacts and validate files already present in the cache."""
    root = cache_root or default_cache_root()
    records = []
    for artifact in spec.artifacts:
        path = artifact_path(spec, artifact, root)
        present = path.is_file()
        actual = sha256_file(path) if present else None
        records.append(
            {
                "name": artifact.name,
                "url": artifact.url,
                "revision": artifact.revision,
                "sha256": artifact.sha256,
                "filename": artifact.filename,
                "required": artifact.required,
                "path": str(path),
                "present": present,
                "verified": present and actual == artifact.sha256,
                "actual_sha256": actual,
            }
        )
    return records


def require_artifacts(spec: ModelSpec, cache_root: Path | None = None) -> dict[str, Path]:
    """Return verified artifact paths or fail before model construction."""
    records = artifact_status(spec, cache_root)
    failures = [record for record in records if record["required"] and not record["verified"]]
    if failures:
        names = ", ".join(str(record["name"]) for record in failures)
        raise FileNotFoundError(
            f"model {spec.name!r} requires missing or invalid artifact(s): {names}. "
            f"Inspect with `tsf model artifacts {spec.name}` and fetch each artifact "
            "explicitly with `--fetch <name>`."
        )
    return {str(record["name"]): Path(str(record["path"])) for record in records if record["verified"]}


def fetch_artifact(
    spec: ModelSpec,
    artifact: ModelArtifact,
    cache_root: Path | None = None,
) -> Path:
    """Download one explicitly requested artifact and atomically verify it.

    ``hf://`` URIs resolve to their pinned Hugging Face download URL.
    """
    root = cache_root or default_cache_root()
    destination = artifact_path(spec, artifact, root)
    try:
        return download(artifact.url, destination, artifact.sha256)
    except ValueError as exc:
        raise ValueError(f"artifact {artifact.name!r} {exc}") from exc

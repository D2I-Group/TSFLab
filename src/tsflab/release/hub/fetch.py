"""Download pinned hub files into the shared TSFLab cache (standard library only)."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from tsflab.release.hub.uri import DEFAULT_ENDPOINT, HubURI, is_hub_uri, parse


def default_cache_root() -> Path:
    """Return the artifact cache (``$TSFLAB_CACHE`` or ``~/.cache/tsflab``)."""
    override = os.environ.get("TSFLAB_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "tsflab"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_url(url: str) -> str:
    """Map ``hf://`` to its pinned HTTPS URL; pass other URLs through."""
    if not is_hub_uri(url):
        return url
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT)
    return parse(url).resolve_url(endpoint)


def hf_token() -> str | None:
    """``HF_TOKEN``, else the token saved by ``hf auth login`` (``HF_HOME``)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    try:
        return (home / "token").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def download(url: str, destination: Path, sha256: str | None = None) -> Path:
    """Atomically download ``url`` to ``destination``, verifying ``sha256`` if given.

    The Hugging Face token (``HF_TOKEN``, else the ``hf auth login`` token file)
    is sent only to the Hugging Face endpoint, so private repositories work
    without exposing the token to other hosts.
    """
    if sha256 and destination.is_file() and sha256_file(destination) == sha256:
        return destination
    resolved = resolve_url(url)
    request = Request(resolved)
    token = hf_token()
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    if token and resolved.startswith(endpoint + "/"):
        request.add_header("Authorization", f"Bearer {token}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with urlopen(request, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        if sha256:
            actual = sha256_file(temporary)
            if actual != sha256:
                raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def cache_path(uri: HubURI, cache_root: Path | None = None) -> Path:
    """Return the cache location of one hub file, keyed by repo and revision."""
    root = cache_root or default_cache_root()
    return root / "hub" / uri.repo_type / uri.repo_id / uri.revision / uri.path


def fetch(uri: str | HubURI, sha256: str | None = None, cache_root: Path | None = None) -> Path:
    """Return a local path for a pinned hub file, downloading it once."""
    parsed = parse(uri) if isinstance(uri, str) else uri
    destination = cache_path(parsed, cache_root)
    if sha256 is None and destination.is_file():
        return destination
    return download(str(parsed), destination, sha256)

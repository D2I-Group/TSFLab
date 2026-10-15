"""Parse ``hf://`` URIs, the one address format for published TSFLab assets.

Grammar (mirrors the Hugging Face filesystem convention)::

    hf://[datasets/|spaces/]<owner>/<repo>@<revision>/<path>

``revision`` is required: every published artifact is pinned to an immutable
commit or tag so the same URI always resolves to the same bytes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import quote

SCHEME = "hf://"
DEFAULT_ENDPOINT = "https://huggingface.co"
# Namespace of the published repositories; a fork or personal mirror sets
# TSFLAB_HUB_OWNER instead of passing --repo everywhere.
DEFAULT_OWNER = os.environ.get("TSFLAB_HUB_OWNER", "Diaugeia")


def default_repo(name: str) -> str:
    """Return ``<owner>/<name>`` in the configured publishing namespace."""
    return f"{DEFAULT_OWNER}/{name}"


_REPO_TYPES = {"datasets": "dataset", "spaces": "space"}
_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_URI = re.compile(
    rf"^hf://(?:(?P<kind>datasets|spaces)/)?(?P<owner>{_NAME})/(?P<repo>{_NAME})"
    rf"@(?P<revision>[A-Za-z0-9._-]+)/(?P<path>[^?#]+)$"
)


@dataclass(frozen=True)
class HubURI:
    repo_id: str
    revision: str
    path: str
    repo_type: str = "model"

    def __str__(self) -> str:
        prefix = {"dataset": "datasets/", "space": "spaces/"}.get(self.repo_type, "")
        return f"{SCHEME}{prefix}{self.repo_id}@{self.revision}/{self.path}"

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def resolve_url(self, endpoint: str = DEFAULT_ENDPOINT) -> str:
        """Return the pinned HTTPS download URL for this file."""
        prefix = {"dataset": "datasets/", "space": "spaces/"}.get(self.repo_type, "")
        return (
            f"{endpoint.rstrip('/')}/{prefix}{self.repo_id}/resolve/{quote(self.revision, safe='')}/{quote(self.path)}"
        )


def is_hub_uri(value: str) -> bool:
    return value.startswith(SCHEME)


def parse(value: str) -> HubURI:
    """Parse and validate one ``hf://`` URI."""
    match = _URI.match(value)
    if match is None:
        raise ValueError(
            f"invalid hub URI {value!r}; expected hf://[datasets/|spaces/]<owner>/<repo>@<revision>/<path>"
        )
    path = match["path"]
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"hub URI path must be a plain relative path: {value!r}")
    return HubURI(
        repo_id=f"{match['owner']}/{match['repo']}",
        revision=match["revision"],
        path=path,
        repo_type=_REPO_TYPES.get(match["kind"] or "", "model"),
    )

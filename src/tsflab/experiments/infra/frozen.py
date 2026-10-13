"""Derive checkpoint-only evaluation from a successful immutable run snapshot."""
from pathlib import Path
import hashlib
import json
from tsflab.experiments.config.loader import LoadedConfig
from tsflab.experiments.config.schema.root import RootConfig


def frozen_evaluation(directory: str, split: str) -> LoadedConfig:
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    root = Path(directory).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "succeeded":
        raise ValueError("frozen evaluation requires a successful source run")
    raw = manifest["config"]
    if raw.get("evaluation", {}).get("checkpoint_path"):
        raise ValueError("source must be a training run, not another evaluation")
    checkpoint = root / "checkpoints" / "best_checkpoint.pth"
    if not checkpoint.is_file():
        raise ValueError("source run has no best checkpoint")
    raw.setdefault("evaluation", {}).update(
        split=split, checkpoint_path=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    config = RootConfig(**raw)
    from tsflab.catalog.registry.datasets import register_dataset_by_name
    register_dataset_by_name(config.dataset.name)
    if manifest.get("data"):
        from tsflab.experiments.infra.fingerprints import dataset_fingerprint
        if dataset_fingerprint(config) != manifest["data"]:
            raise ValueError("source dataset changed since training")
    return LoadedConfig(raw=raw, config=config, sweep_keys=[], config_name="frozen-evaluation")

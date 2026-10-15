"""Read-only environment audit and shared experiment preflight checks."""

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import sys

from tsflab.experiments.infra.policy import ExecutionPolicy
from tsflab.experiments.infra.hardware import gpu_inventory


def _check_packages(check) -> dict:
    """Check the installed versions of tsflab's core requirements; return the found versions."""
    packages = {}
    from packaging.requirements import Requirement

    requirements = importlib.metadata.requires("tsflab") or []
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = requirement.name
        try:
            version = importlib.metadata.version(name)
            packages[name] = version
            check(
                name,
                not requirement.specifier or version in requirement.specifier,
                f"{version}; required {requirement.specifier or 'any'}",
            )
        except importlib.metadata.PackageNotFoundError:
            check(
                name, False, "package is missing; synchronize the project environment"
            )
    return packages


def _check_tracking(policy, check) -> None:
    for enabled, name in (
        (policy.tracking.tensorboard, "tensorboard"),
        (policy.tracking.wandb != "disabled", "wandb"),
    ):
        if enabled:
            check(
                name,
                importlib.util.find_spec(name) is not None,
                f"optional dependency: install tsflab[{name}]",
            )


def _check_accelerators(policy, device, check) -> None:
    try:
        import torch

        if device == "cuda" or policy.resources.gpus:
            check(
                "cuda",
                torch.cuda.is_available(),
                f"torch CUDA build: {torch.version.cuda}",
            )
        if device == "mps":
            check(
                "mps", torch.backends.mps.is_available(), "requested Apple accelerator"
            )
    except Exception as exc:
        check("torch-runtime", False, str(exc))


def _check_output(policy, work_dir, check) -> None:
    path = Path(work_dir).expanduser().resolve()
    while not path.exists():
        path = path.parent
    free_gb = shutil.disk_usage(path).free / 1024**3
    check("output-writable", os.access(path, os.W_OK), str(path))
    check(
        "disk", free_gb >= policy.resources.min_free_disk_gb, f"{free_gb:.2f} GiB free"
    )


def _check_gpu_list(policy, devices, check) -> None:
    """Check the configured GPUs resolve to distinct visible devices and cover ``gpus_per_run``."""
    resolved_ids = [
        g["uuid"]
        for selected in policy.resources.gpus
        for g in devices
        if selected in (g["index"], g["uuid"])
    ]
    if len(resolved_ids) != len(set(resolved_ids)) or len(policy.resources.gpus) != len(
        set(policy.resources.gpus)
    ):
        check("gpu-list", False, "GPU identifiers must be unique")
    if policy.resources.gpus and policy.resources.gpus_per_run > len(
        policy.resources.gpus
    ):
        check("gpu-count", False, "gpus_per_run exceeds available configured GPUs")


def audit_environment(
    policy: ExecutionPolicy | None = None,
    *,
    device: str | None = None,
    work_dir="work_dirs",
) -> dict:
    """Check dependencies, requested accelerators, and output capacity; never install."""
    policy = policy or ExecutionPolicy()
    checks = []

    def check(name, ok, detail):
        checks.append(
            {"name": name, "status": "passed" if ok else "failed", "detail": detail}
        )

    check("python", sys.version_info >= (3, 12), sys.version.split()[0])
    packages = _check_packages(check)
    _check_tracking(policy, check)
    _check_accelerators(policy, device, check)
    _check_output(policy, work_dir, check)
    devices = gpu_inventory()
    for selected in policy.resources.gpus:
        check(
            f"gpu:{selected}",
            any(selected in (g["index"], g["uuid"]) for g in devices),
            "requested NVIDIA device",
        )
    _check_gpu_list(policy, devices, check)
    return {
        "schema_version": 1,
        "ok": all(c["status"] == "passed" for c in checks),
        "checks": checks,
        "gpus": devices,
        "packages": packages,
    }


def _validate_training(spec, config, required, loss) -> None:
    """Check the output/loss pairing, optimizer, and loss of a trainable model."""
    if (
        (required and loss != required)
        or (not required and loss in {"quantile", "nll_gaussian"})
    ):
        raise ValueError(f"incompatible output/loss: {spec.output_type}/{loss}")
    import torch
    from tsflab.catalog.registry.losses import get_loss

    if not hasattr(torch.optim, config.training.optimizer.name):
        raise ValueError(f"unknown optimizer: {config.training.optimizer.name}")
    # Construction may require quantile levels, which the trainer resolves later.
    params = dict(config.training.loss_params)
    if loss == "quantile":
        params.setdefault(
            "quantile_levels", list(config.evaluation.quantile_levels)
        )
    get_loss(config.training.loss, **params)


def validate_experiment(config) -> None:
    """Check static runtime compatibility before loading data or allocating a model."""
    from tsflab.catalog.registry.models import MODEL_CATALOG
    from tsflab.catalog.registry.datasets import DATASET_REGISTRY

    spec = MODEL_CATALOG.get(config.model.name)
    data = DATASET_REGISTRY.get(config.dataset.name)
    data.resolve_location(config.dataset.path, config.dataset.id)
    if config.dataset.path and not Path(config.dataset.path).expanduser().exists():
        raise ValueError(f"dataset path does not exist: {config.dataset.path}")
    _validate_csv_channels(config, data)
    if spec.artifacts:
        from tsflab.catalog.model_artifacts import require_artifacts

        require_artifacts(spec)
    inference = "inference-only" in spec.capabilities
    if inference and config.experiment.runtime.use_multi_gpu:
        raise ValueError("inference-only models do not support DataParallel")
    required = {"quantile": "quantile", "distribution": "nll_gaussian"}.get(
        spec.output_type
    )
    loss = config.training.loss.lower()
    if not inference:
        _validate_training(spec, config, required, loss)


def _validate_csv_channels(config, data):
    """Inspect known flat-CSV headers without constructing loaders or models."""
    if config.task.mode != "time_series" or "enc_in" not in config.model.params:
        return
    from tsflab.data.datasets.custom import Dataset_Custom
    from tsflab.data.datasets.ett import Dataset_ETT_hour, Dataset_ETT_minute

    if getattr(data, "dataset_class", None) not in (Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute):
        return
    import csv

    path = Path(config.dataset.path).expanduser()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream), [])
    params = config.dataset.params
    if hasattr(params, "model_dump"):
        params = params.model_dump()
    target = params.get("target", "OT")
    if len(columns) < 2 or target not in columns[1:]:
        raise ValueError(f"dataset CSV must contain timestamp and target column {target!r}: {path}")
    channels = 1 if config.task.features == "S" else len(columns) - 1
    configured = config.model.params["enc_in"]
    if configured != channels:
        raise ValueError(
            f"{config.model.name}: model.params.enc_in={configured} does not match "
            f"{channels} dataset input channels (features={config.task.features}); "
            f"set model.params.enc_in={channels} and compatible model channel fields"
        )

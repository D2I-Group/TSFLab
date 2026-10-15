"""Host-local GPU leases with free-memory checks and crash-safe ownership."""

from contextlib import ExitStack, contextmanager
import hashlib
from itertools import combinations
import os
from pathlib import Path
import time

from tsflab.experiments.infra.hardware import gpu_inventory, validate_gpu_visibility
from tsflab.experiments.infra.storage import file_lock


class GPUAssignment(list):
    def __init__(self, items, descriptors=()):
        super().__init__(items)
        self.descriptors = tuple(descriptors)


def _resource_root(directory) -> Path:
    """Directory of the host-wide GPU lock and reservation files."""
    return Path(
        directory
        if directory is not None
        else os.environ.get(
            "TSFLAB_RESOURCE_DIR", Path.home() / ".cache" / "tsflab" / "resources"
        )
    )


def _eligible_gpus(devices, resources) -> list:
    """Configured GPUs with enough free memory (and idle compute under auto-selection)."""
    return [
        gpu
        for gpu in devices
        if any(item in (gpu["index"], gpu["uuid"]) for item in resources.gpus)
        and float(gpu["free_mb"]) >= resources.min_free_memory_mb
        and (not resources.auto_select or gpu.get("compute_idle") is True)
    ]


def _lock_group(group, root):
    """Take the exclusive lock of every GPU in ``group``; return (stack, descriptors), or None if one is held."""
    stack = ExitStack()
    descriptors = []
    try:
        for gpu in sorted(group, key=lambda g: g["uuid"]):
            key = hashlib.sha256(gpu["uuid"].encode()).hexdigest()
            descriptors.append(
                stack.enter_context(
                    file_lock(root / f"{key}.lock", blocking=False)
                )
            )
    except BlockingIOError:
        stack.close()
        return None
    return stack, descriptors


def _group_still_fits(group, query, resources) -> bool:
    """Whether every GPU of a locked group still has the memory (and idle compute) it was chosen for."""
    # Another process may have consumed memory while locks were acquired.
    current = {gpu["uuid"]: gpu for gpu in query()}
    return not any(
        float(current.get(gpu["uuid"], {}).get("free_mb", -1)) < resources.min_free_memory_mb
        or (resources.auto_select and current.get(gpu["uuid"], {}).get("compute_idle") is not True)
        for gpu in group
    )


@contextmanager
def _exclusive_gpus(resources, *, cancelled, query, directory):
    root = _resource_root(directory)
    deadline = time.monotonic() + resources.wait_timeout_minutes * 60
    while time.monotonic() < deadline:
        if cancelled():
            raise InterruptedError("GPU wait cancelled or budget expired")
        devices = query()
        validate_gpu_visibility(resources.gpus, devices)
        eligible = _eligible_gpus(devices, resources)
        for group in combinations(eligible, resources.gpus_per_run):
            locked = _lock_group(group, root)
            if locked is None:
                continue
            stack, descriptors = locked
            try:
                if not _group_still_fits(group, query, resources):
                    continue
                yield GPUAssignment([gpu["uuid"] for gpu in group], descriptors)
                return
            finally:
                stack.close()
        time.sleep(0.2)
    raise TimeoutError("GPU resource wait exceeded policy timeout")


@contextmanager
def lease_gpus(
    resources=None, *, cancelled=lambda: False, inventory=None, directory=None
):
    from tsflab.experiments.infra.policy import Resources

    resources = resources if resources is not None else Resources()
    query = inventory if inventory is not None else gpu_inventory
    if resources.sharing and resources.gpus:
        with shared_gpus(
            resources, cancelled=cancelled, inventory=query, directory=directory
        ) as assigned:
            yield assigned
        return
    if not resources.gpus:
        yield []
        return
    with _exclusive_gpus(resources, cancelled=cancelled, query=query, directory=directory) as assigned:
        yield assigned


def _live_reservations(root) -> list:
    """Reservations whose owners still hold their locks; stale reservation files are removed."""
    import json

    reservations = []
    for path in root.glob("reservation-*.json"):
        try:
            with file_lock(path.with_suffix(".lock"), blocking=False):
                path.unlink()
        except BlockingIOError:
            reservations.append(json.loads(path.read_text()))
    return reservations


def _has_shared_capacity(gpu, reservations, resources) -> bool:
    """Whether a configured GPU has a process slot and memory left for one more shared run."""
    if not any(
        item in (gpu["index"], gpu["uuid"]) for item in resources.gpus
    ):
        return False
    used = [r for r in reservations if gpu["uuid"] in r["gpus"]]
    if len(used) >= resources.max_processes_per_gpu or sum(
        r["memory_mb"] for r in used
    ) + resources.memory_per_run_mb > float(gpu["total_mb"]):
        return False
    return not float(gpu["free_mb"]) < max(
        resources.memory_per_run_mb, resources.min_free_memory_mb
    )


def _share_devices(devices, reservations, resources, root, stack, descriptors, selected) -> None:
    """Take shared locks on GPUs with capacity until ``gpus_per_run`` are selected."""
    import fcntl

    for gpu in devices:
        if not _has_shared_capacity(gpu, reservations, resources):
            continue
        key = hashlib.sha256(gpu["uuid"].encode()).hexdigest()
        stream = (root / f"{key}.lock").open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            continue
        stack.callback(stream.close)
        descriptors.append(stream.fileno())
        selected.append(gpu["uuid"])
        if len(selected) == resources.gpus_per_run:
            break


@contextmanager
def shared_gpus(resources, *, cancelled, inventory=None, directory=None):
    query = inventory if inventory is not None else gpu_inventory
    """Cooperative memory reservations; does not isolate GPU memory physically."""
    import uuid
    from tsflab.experiments.infra.storage import write_json

    if resources.memory_per_run_mb <= 0:
        raise ValueError("shared GPUs require memory_per_run_mb > 0")
    root = _resource_root(directory)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + resources.wait_timeout_minutes * 60
    while time.monotonic() < deadline:
        if cancelled():
            raise InterruptedError("shared GPU wait cancelled")
        stack = ExitStack()
        selected = []
        descriptors = []
        receipt = None
        try:
            with file_lock(root / ".sharing.lock"):
                reservations = _live_reservations(root)
                devices = query()
                validate_gpu_visibility(resources.gpus, devices)
                _share_devices(devices, reservations, resources, root, stack, descriptors, selected)
                if len(selected) == resources.gpus_per_run:
                    receipt = root / f"reservation-{uuid.uuid4().hex}.json"
                    descriptors.append(
                        stack.enter_context(
                            file_lock(receipt.with_suffix(".lock"), blocking=False)
                        )
                    )
                    write_json(
                        receipt,
                        {"gpus": selected, "memory_mb": resources.memory_per_run_mb},
                    )
            if receipt is not None:
                yield GPUAssignment(selected, descriptors)
                return
        finally:
            if receipt is not None:
                with file_lock(root / ".sharing.lock"):
                    receipt.unlink(missing_ok=True)
            stack.close()
        time.sleep(0.2)
    raise TimeoutError("shared GPU wait exceeded policy timeout")

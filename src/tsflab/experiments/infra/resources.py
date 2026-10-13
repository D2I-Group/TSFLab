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
    root = Path(
        directory
        if directory is not None
        else os.environ.get(
            "TSFLAB_RESOURCE_DIR", Path.home() / ".cache" / "tsflab" / "resources"
        )
    )
    deadline = time.monotonic() + resources.wait_timeout_minutes * 60
    while time.monotonic() < deadline:
        if cancelled():
            raise InterruptedError("GPU wait cancelled or budget expired")
        devices = query()
        validate_gpu_visibility(resources.gpus, devices)
        eligible = [
            gpu
            for gpu in devices
            if any(item in (gpu["index"], gpu["uuid"]) for item in resources.gpus)
            and float(gpu["free_mb"]) >= resources.min_free_memory_mb
            and (not resources.auto_select or gpu.get("compute_idle") is True)
        ]
        for group in combinations(eligible, resources.gpus_per_run):
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
                continue
            try:
                # Another process may have consumed memory while locks were acquired.
                current = {gpu["uuid"]: gpu for gpu in query()}
                if any(
                    float(current.get(gpu["uuid"], {}).get("free_mb", -1)) < resources.min_free_memory_mb
                    or (resources.auto_select and current.get(gpu["uuid"], {}).get("compute_idle") is not True)
                    for gpu in group
                ):
                    continue
                yield GPUAssignment([gpu["uuid"] for gpu in group], descriptors)
                return
            finally:
                stack.close()
        time.sleep(0.2)
    raise TimeoutError("GPU resource wait exceeded policy timeout")


@contextmanager
def shared_gpus(resources, *, cancelled, inventory=None, directory=None):
    query = inventory if inventory is not None else gpu_inventory
    """Cooperative memory reservations; does not isolate GPU memory physically."""
    import fcntl
    import json
    import uuid
    from tsflab.experiments.infra.storage import write_json

    if resources.memory_per_run_mb <= 0:
        raise ValueError("shared GPUs require memory_per_run_mb > 0")
    root = Path(
        directory
        if directory is not None
        else os.environ.get(
            "TSFLAB_RESOURCE_DIR", Path.home() / ".cache" / "tsflab" / "resources"
        )
    )
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
                reservations = []
                for path in root.glob("reservation-*.json"):
                    try:
                        with file_lock(path.with_suffix(".lock"), blocking=False):
                            path.unlink()
                    except BlockingIOError:
                        reservations.append(json.loads(path.read_text()))
                devices = query()
                validate_gpu_visibility(resources.gpus, devices)
                for gpu in devices:
                    if not any(
                        item in (gpu["index"], gpu["uuid"]) for item in resources.gpus
                    ):
                        continue
                    used = [r for r in reservations if gpu["uuid"] in r["gpus"]]
                    if len(used) >= resources.max_processes_per_gpu or sum(
                        r["memory_mb"] for r in used
                    ) + resources.memory_per_run_mb > float(gpu["total_mb"]):
                        continue
                    if float(gpu["free_mb"]) < max(
                        resources.memory_per_run_mb, resources.min_free_memory_mb
                    ):
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

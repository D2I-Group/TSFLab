"""Read-only accelerator inventory, without experiment or audit dependencies."""

import subprocess
import os


def validate_gpu_visibility(selected, inventory):
    """Explicit GPU identifiers are physical indices/UUIDs, within the parent mask."""
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return
    tokens = {item.strip() for item in mask.split(",") if item.strip()}
    allowed = set(tokens)
    for gpu in inventory:
        if tokens.intersection((gpu["index"], gpu["uuid"])):
            allowed.update((gpu["index"], gpu["uuid"]))
    outside = [item for item in selected if item not in allowed]
    if outside:
        raise ValueError(
            f"GPU identifiers {outside} are outside CUDA_VISIBLE_DEVICES={mask!r}; "
            "use --gpus auto or visible physical NVIDIA indices/UUIDs"
        )


def gpu_inventory() -> list[dict]:
    """Query NVIDIA metadata without importing a training runtime."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        busy = set(processes.stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        busy = None
    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            devices.append(
                dict(zip(("index", "uuid", "name", "free_mb", "total_mb"), fields))
            )
            devices[-1]["compute_idle"] = None if busy is None else fields[1] not in busy
    return devices

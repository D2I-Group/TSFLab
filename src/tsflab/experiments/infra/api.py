"""Public, lazy Python facade for independently composable infrastructure services.

Importing this facade neither initializes services nor imports optional runtimes.
Domain modules remain available for focused imports; this table defines the
supported facade and powers CLI discovery without importing implementations.
"""

from importlib import import_module

# Export name -> implementation module. No parallel implementations or registry
# of scientific models are maintained here.
_EXPORTS = {
    "invoke": "results",
    "OperationResult": "results",
    "ContractError": "results",
    "resolve_policy": "execution",
    "load_executor": "executors",
    "prepare_task": "research",
    "load_task": "research",
    "create_round": "research",
    "load_round": "research",
    "read_events": "research",
    "add_event": "research",
    "set_round_status": "research",
    "claim_iteration": "research",
    "ExecutionPolicy": "policy",
    "Budget": "policy",
    "Resources": "policy",
    "Recovery": "policy",
    "Storage": "policy",
    "Tracking": "policy",
    "load_policy": "policy",
    "Cancellation": "contracts",
    "Executor": "contracts",
    "MetricsSink": "contracts",
    "RuntimeState": "contracts",
    "FileCancellation": "contracts",
    "any_cancelled": "contracts",
    "UsageLedger": "accounting",
    "account": "accounting",
    "gpu_inventory": "hardware",
    "lease_gpus": "resources",
    "Tracker": "tracking",
    "storage_status": "retention",
    "cleanup": "retention",
    "audit_environment": "environment",
    "validate_experiment": "environment",
    "canonical_hash": "storage",
    "write_json": "storage",
    "file_lock": "storage",
    "code_fingerprint": "fingerprints",
    "dataset_fingerprint": "fingerprints",
    "dependency_fingerprint": "fingerprints",
    "save_checkpoint": "checkpoint",
    "restore_checkpoint": "checkpoint",
    "runtime_state": "checkpoint",
    "restore_runtime_state": "checkpoint",
    "compare_rows": "comparison",
    "protocol_fingerprint": "comparison",
    "enqueue": "queue",
    "jobs": "queue",
    "cancel_job": "queue",
    "run_job": "queue",
    "work": "queue",
    "slurm": "slurm",
    "preflight": "execution",
    "prepare_sweep": "execution",
    "execute": "execution",
    "status": "execution",
    "wait": "execution",
    "cancel": "execution",
}

# Requirements describe operational boundaries, not import-time dependencies.
_REQUIREMENTS = {
    "research": "Optional templates and persistent budget/evidence records; never plans or dispatches an Agent",
    "checkpoint": "PyTorch model/optimizer; trusted local state files",
    "environment": "Runtime/data checks import or inspect requested dependencies",
    "execution": "Resolved experiment configuration and managed run manifests",
    "fingerprints": "Code fingerprint uses repository/installed-package context",
    "queue": "POSIX locks; detached work uses the built-in sweep executor; run_job accepts an executor",
    "resources": "POSIX locks; CUDA allocation uses NVIDIA inventory or an injected provider",
    "retention": "Inspection accepts any existing directory; cleanup requires a managed run and may load PyTorch state",
    "slurm": "Slurm CLI and shared filesystem; submission is explicit",
    "tracking": "Local JSONL only by default; requested mirrors require their optional packages",
    "storage": "Local filesystem; file_lock requires POSIX",
}

__all__ = sorted([*_EXPORTS, "describe_modules"])


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"tsflab.experiments.infra.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def describe_modules():
    """Describe callable groups without loading them or probing external systems."""
    return {
        "schema_version": 1,
        "python_api": "tsflab.experiments.infra.api",
        "modules": [
            {
                "name": module,
                "import": f"tsflab.experiments.infra.{module}",
                "exports": sorted(
                    name for name, owner in _EXPORTS.items() if owner == module
                ),
                "requirements": _REQUIREMENTS.get(
                    module,
                    "No experiment, training runtime, or external service required",
                ),
            }
            for module in sorted(set(_EXPORTS.values()))
        ],
    }

"""Static driver -> install/run profile mapping; no torch device is touched."""

from pathlib import Path

import pytest

from tsflab.experiments.infra.machine import (
    RUN_PROFILES,
    cuda_from_driver,
    install_commands,
    install_profile,
    parse_cuda_version,
    run_profile,
    torch_build_tag,
)
from tsflab.experiments.infra.policy import load_policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("cuda", "expected"),
    [
        (None, "cpu"),
        ("12.4", "cpu"),  # too old for any torch 2.14.1 CUDA wheel
        ("12.6", "cuda126"),
        ("12.8", "cuda126"),  # driver 570: no cu128 wheel for torch 2.14.1
        ("12.9", "cuda126"),
        ("13.0", "cuda130"),
        ("13.4", "cuda130"),  # driver 615 (RTX 50-series)
    ],
)
def test_driver_cuda_maps_to_install_profile(cuda, expected):
    assert install_profile(parse_cuda_version(cuda), "Linux") == expected


def test_non_linux_always_uses_cpu_profile():
    assert install_profile((13, 0), "Darwin") == "cpu"
    assert install_profile((12, 8), "Windows") == "cpu"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("| NVIDIA-SMI 570.211.01   Driver Version: 570.211.01   CUDA Version: 12.8 |", (12, 8)),
        ("| NVIDIA-SMI 615.71.09   KMD Version: 615.71.09   CUDA UMD Version: 13.4 |", (13, 4)),
        ("no gpu", None),
    ],
)
def test_parse_nvidia_smi_header(header, expected):
    assert parse_cuda_version(header) == expected


def test_driver_major_fallback():
    assert cuda_from_driver("570.211.01") == (12, 8)
    assert cuda_from_driver("615.71.09") == (13, 0)
    assert cuda_from_driver("560.35.03") == (12, 6)
    assert cuda_from_driver("unknown") is None


def test_install_commands_use_pip_backend_only_where_sync_cannot():
    assert install_commands("cuda130") == [
        "uv sync --frozen --python 3.12 --extra models --extra data --extra experiments"
    ]
    cu126 = install_commands("cuda126")
    assert cu126[0] == "uv venv --python 3.12"
    assert "--torch-backend cu126" in cu126[1]
    assert any("UV_NO_SYNC=1" in line for line in cu126)
    assert "--torch-backend cpu" in install_commands("cpu", "Linux")[1]
    assert install_commands("cpu", "Darwin")[0].startswith("uv sync --frozen")


def test_run_profiles_and_build_tags():
    assert [run_profile(n) for n in (0, 1, 8)] == ["laptop-cpu", "single-gpu", "multi-gpu"]
    assert torch_build_tag("12.6") == "cu126" and torch_build_tag(None) == "cpu"


@pytest.mark.parametrize("name", sorted(RUN_PROFILES))
def test_run_profile_policies_validate(name):
    policy = load_policy(str(ROOT / RUN_PROFILES[name]))
    assert policy.resources.threads_per_run
    if policy.resources.sharing:
        assert policy.resources.memory_per_run_mb > 0


def test_setup_installs_the_driver_profile_and_checks_torch(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    from tsflab.experiments.infra import machine

    monkeypatch.setattr(machine, "_nvidia", lambda: {"driver": "570.86", "cuda": "12.8"})
    monkeypatch.setattr(machine.platform, "system", lambda: "Linux")
    calls = []

    def run(command, **kw):
        calls.append(command)
        if isinstance(command, list):  # the torch probe
            return NS(returncode=0, stdout='{"version": "2.14.1", "cuda": "12.6", "cuda_available": true}\n')
        return NS(returncode=0, stdout="")

    plan = machine.setup(str(tmp_path), dry_run=True, run=run)
    assert plan["profile"] == "cuda126" and not calls
    assert any("--torch-backend cu126" in c for c in plan["commands"])
    report = machine.setup(str(tmp_path), run=run)
    assert report["ok"] and report["no_sync"] and report["torch"]["cuda_available"]
    assert calls[0] == "uv venv --python 3.12" and isinstance(calls[-1], list)
    assert "export" not in " ".join(c for c in calls if isinstance(c, str))
    failing = machine.setup(str(tmp_path), run=lambda c, **kw: NS(returncode=1, stdout=""))
    assert not failing["ok"] and failing["failed"] == "uv venv --python 3.12"

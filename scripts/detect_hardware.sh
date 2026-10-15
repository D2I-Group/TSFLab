#!/usr/bin/env bash
#
# Pre-install hardware probe: map the NVIDIA driver to a TSFLab install profile
# for torch 2.14.1 (cpu | cuda126 | cuda130). Same mapping as `tsf env doctor`
# (src/tsflab/experiments/infra/machine.py), which also checks the installed
# torch build once the environment exists.
#
#   driver CUDA >= 13.0        -> cuda130  (uv sync --frozen; the locked build)
#   driver CUDA 12.6 .. 12.9   -> cuda126  (torch 2.14.1 has no cu128 wheel)
#   older driver / no NVIDIA   -> cpu
#
# Usage:
#   bash scripts/detect_hardware.sh             # human-readable report
#   bash scripts/detect_hardware.sh --backend   # print only the uv backend tag
#   bash scripts/detect_hardware.sh --profile   # print only the profile name
set -euo pipefail

MODE="${1:-report}"
gpu="none"; driver="none"; cuda=""

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd';' - || true)"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  # Max CUDA version the driver supports: "CUDA Version: 12.8" or "CUDA UMD Version: 13.4".
  cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA[A-Za-z ]*Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"
  if [ -z "$cuda" ]; then  # fall back to the driver branch
    major="${driver%%.*}"
    if [ "${major:-0}" -ge 580 ] 2>/dev/null; then cuda="13.0"
    elif [ "${major:-0}" -ge 560 ] 2>/dev/null; then cuda="12.6"
    fi
  fi
fi

profile="cpu"
if [ "$(uname -s)" = "Linux" ] && [ -n "$cuda" ]; then
  cmajor="${cuda%%.*}"; cminor="${cuda#*.}"
  if [ "$cmajor" -ge 13 ]; then profile="cuda130"
  elif [ "$cmajor" -eq 12 ] && [ "$cminor" -ge 6 ]; then profile="cuda126"
  fi
fi
case "$profile" in
  cuda130) backend="cu130" ;;
  cuda126) backend="cu126" ;;
  *)       backend="cpu" ;;
esac

case "$MODE" in
  --backend) echo "$backend" ;;
  --profile) echo "$profile" ;;
  *)
    echo "gpu=${gpu:-unknown}"
    echo "driver=${driver:-unknown}"
    echo "cuda=${cuda:-none}"
    echo "profile=${profile}"
    echo "backend=${backend}"
    if [ "$profile" = "cuda130" ] || [ "$(uname -s)" != "Linux" ]; then
      echo "install: uv sync --frozen --python 3.12 --extra models --extra data --extra experiments"
    else
      echo "install: uv venv --python 3.12 && uv pip install --torch-backend ${backend} -e \".[models,data,experiments]\" pytest && export UV_NO_SYNC=1"
    fi
    ;;
esac

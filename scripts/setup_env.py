"""Set up the TSFLab environment for this machine: python3 scripts/setup_env.py [--dry-run] [--extras a,b].

Detects the NVIDIA driver, installs the torch build it supports (cu130, cu126, or cpu) with
uv, and checks that torch sees the GPU. Runs before TSFLab is installed: it loads the
standard-library module ``tsflab/experiments/infra/machine.py`` by path.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the detected profile and commands only")
    parser.add_argument("--extras", default="models,data,experiments", help="comma-separated optional extras")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("machine", ROOT / "src/tsflab/experiments/infra/machine.py")
    machine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(machine)
    extras = tuple(e for e in args.extras.split(",") if e)
    report = machine.setup(str(ROOT), extras, args.dry_run)
    print(machine.format_setup(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

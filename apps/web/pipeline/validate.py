#!/usr/bin/env python3
"""Validate every submission in submissions/ against the TSF-Core contract.

The contract is the pydantic models in ``tsflab.core`` (RunRecord for
flat records, SubmissionReport for bundles); only ``pydantic`` is needed.
Exit code is non-zero if anything fails, so CI can gate PRs.

Usage:  python3 pipeline/validate.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from tsflab.core.leaderboard import load_submissions  # noqa: E402


def main() -> int:
    valid, rejected = load_submissions(ROOT / "submissions")
    for path, errors in rejected.items():
        print(f"✗ {path.relative_to(ROOT)}")
        for error in errors[:5]:
            print(f"    - {error}")
    total = len(valid) + len(rejected)
    print(f"\n{len(valid)}/{total} submissions valid.")
    if rejected:
        print(f"❌ {len(rejected)} invalid submission(s).")
        return 1
    print("✅ all submissions valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

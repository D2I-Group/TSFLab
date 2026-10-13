#!/usr/bin/env bash
# Package one session: diff against the baseline commit (including small code and config
# files that the codebase's .gitignore hides, e.g. under experiments/ or work_dirs/),
# then check success and compute metrics. package.sh <task.json> <run-dir>
set -euo pipefail
TASK="$(realpath "$1")"; RUN="$(realpath "$2")"; HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$RUN/workspace"
git add -A
git ls-files --others --ignored --exclude-standard -z \
  | python3 -c '
import os, re, sys
keep = re.compile(r"\.(py|sh|toml|ya?ml|json|md|txt|cfg|ini|ipynb)$")
skip = re.compile(r"(^|/)(\.venv|\.cache|__pycache__|\.git|checkpoints?|wandb|lightning_logs|node_modules)(/|$)")
for p in sys.stdin.buffer.read().decode().split("\0"):
    if p and keep.search(p) and not skip.search(p) and os.path.getsize(p) < 1_000_000:
        sys.stdout.write(p + "\0")
' | xargs -0 -r git add -f --
git diff --cached --numstat HEAD > "$RUN/diff.numstat"
git diff --cached HEAD > "$RUN/diff.patch"
cd - > /dev/null
# Preserve only usage metadata, not credentials or the full private rollout.
# Exports must include this small file to permit later per-call accounting.
python3 - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
files = sorted((run / "codex" / "sessions").rglob("*.jsonl"))
if files:
    with (run / "usage-events.jsonl").open("w") as out:
        for path in files:
            for line in path.open():
                if '"token_count"' not in line:
                    continue
                e = json.loads(line)
                info = (e.get("payload") or {}).get("info") or {}
                if info.get("total_token_usage"):
                    out.write(json.dumps({"timestamp": e["timestamp"], "usage_chain": path.stem, "payload": {
                        "type": "token_count", "info": {k: info[k] for k in
                        ("total_token_usage", "last_token_usage") if k in info}}}) + "\n")
PY
python3 "$HERE/check_success.py" "$TASK" "$RUN/workspace" > "$RUN/success.json" || true
python3 "$HERE/analyze.py" "$RUN" > "$RUN/metrics.json"

"""Auditable core-process ledger; never infer exact tokens from text lengths.

python3 core_stats.py ARCHIVE OUTPUT.json
Uses the final campaign versions and the planned repetitions, equally for all
arms. Supervised attempts and pure monitoring commands are separate records.
Mixed commands, debugging and ordinary experimental search remain core unless
an explicit, evidence-backed exclusion is added in a future protocol.
"""
import json
import argparse
import csv
import hashlib
import re
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

VERSIONS = {"reproduce": "539b6e5e", "benchmark": "5fbcaaf8", "autoresearch": "94fa2e65"}
ARMS = {"tsflab", "tslib", "tfb", "empty"}


def shell_text(command):
    try:
        argv = shlex.split(command)
    except ValueError:
        return command
    if len(argv) == 3 and Path(argv[0]).name in {"bash", "sh", "zsh"} and argv[1] in {"-c", "-lc"}:
        return argv[2]
    return command


def overhead(command, protocol="core-v2"):
    """Conservative: remove only standalone monitoring; mixed commands stay."""
    command = shell_text(command).strip()
    if any(x in command for x in (";", "&&", "||", "|", "\n", ">", "<", "`", "$(")):
        return None
    command = re.sub(r"^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+", "", command)
    if re.fullmatch(r"stage\s+[\w-]+", command):
        return "stage_marker"
    if re.fullmatch(r"(?:sleep\s+[\d.]+[smh]?|nvidia-smi(?:\s+[^;&|<>]*)?|ps\s+[-\w ]+)", command):
        return "monitoring"
    if re.fullmatch(r"(?:uv\s+run\s+)?tsf\s+run\s+status\s+[^;&|<>]+", command):
        return "monitoring"
    if re.fullmatch(r"(?:uv\s+run\s+)?tsf\s+run\s+wait\s+[^;&|<>]+", command):
        # Waiting for a submitted job is execution, not redundant supervision.
        if protocol == "core-v1" or re.search(r"--timeout(?:=|\s+)0(?:\.0*)?(?:\s|$)", command):
            return "monitoring"
        return None
    if re.match(r"^(?:tail|head|cat)\s+", command):
        try:
            args = shlex.split(command)[1:]
        except ValueError:
            return None
        files = [x for x in args if not x.startswith("-") and not x.isdigit()]
        if files and all(x.endswith((".log", "/events.jsonl", "/runtime.json")) for x in files):
            return "log_inspection_candidate"
    return None


def planned(s):
    if s["arm"] not in ARMS or (s["agent"], s["model"]) != ("codex", "gpt-6-luna"):
        return False
    if s["arm"] == "tsflab" and s["version"] != VERSIONS.get(s["kind"]):
        return False
    if s["kind"] == "benchmark" and s["task"] not in {"bench2-ercot", "bench2-weather"}:
        return False
    match = re.search(r"-r(\d+)$", s["name"])
    return bool(match and int(match[1]) <= (2 if s["kind"] == "autoresearch" else 3))


def audit_session(base, s, protocol="core-v2"):
    run = base / "sessions" / s["root"] / s["name"]
    counts = Counter()
    excluded = []
    snapshots = []
    complete = False
    seen_logs = set()
    for stream in sorted(run.glob("stream.*.jsonl")):
        attempt = int(stream.name.split(".")[1])
        for lineno, line in enumerate(stream.open(), 1):
            d = json.loads(line)["d"]
            if attempt == 0 and d.get("type") == "turn.completed":
                u = d.get("usage") or {}
                snapshots.append((u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0))
                complete = True
            if d.get("type") != "item.completed":
                continue
            it = d.get("item") or {}
            if it.get("type") not in {"command_execution", "file_change", "web_search", "mcp_tool_call"}:
                continue
            reason = "supervised_attempt" if attempt else None
            if reason is None and it.get("type") == "command_execution":
                reason = overhead(it.get("command", ""), protocol)
            # A log inspection may be legitimate debugging. Flag, retain it.
            if reason == "log_inspection_candidate":
                command = shell_text(it.get("command", ""))
                output = it.get("aggregated_output", "") or ""
                failed = it.get("exit_code") not in (None, 0) or re.search(
                    r"Traceback|\bERROR\b|RuntimeError|out of memory|CUDNN_STATUS", output, re.I)
                if command in seen_logs and not failed:
                    reason = "repeated_healthy_log_probe"
                else:
                    counts["log_inspection_candidates_retained"] += 1
                    reason = None
                seen_logs.add(command)
            if reason:
                counts[reason] += 1
                excluded.append({"file": str(stream.relative_to(base)), "line": lineno,
                                 "item_id": it.get("id"), "reason": reason})
            else:
                counts["core_calls"] += 1
    return {"root": s["root"], "name": s["name"], "kind": s["kind"], "task": s["task"], "arm": s["arm"],
            "version": s["version"],
            "counts": dict(counts), "excluded_calls": excluded,
            "raw_tokens_m": s["tokens_m"],
            "initial_attempt_tokens_m": max(snapshots) / 1e6 if complete else None,
            "core_tokens_m": None,
            "core_tokens_status": "unidentifiable_without_per_call_usage; initial attempt includes monitoring",
            "first_attempt_outcome": "unknown_after_resume" if s.get("interventions") else
                ("solved" if s["completion"] >= 1 and s["session_valid"] else "unsolved"),
            "final_session_valid": s["session_valid"], "interventions": s.get("interventions", 0)}


def build_report(base, arm=None, protocol="core-v2"):
    base = Path(base)
    sessions = json.loads((base / "sessions.json").read_text())
    selected = [s for s in sessions if planned(s) and not s.get("cudnn") and (arm is None or s["arm"] == arm)]
    rows = [audit_session(base, s, protocol) for s in selected]
    groups = defaultdict(list)
    for row in rows:
        groups[row["kind"], row["arm"]].append(row)
    summary = []
    for (kind, group_arm), group in groups.items():
        c = Counter()
        for row in group:
            c.update(row["counts"])
        values = [r["initial_attempt_tokens_m"] for r in group if r["initial_attempt_tokens_m"] is not None]
        summary.append({"kind": kind, "arm": group_arm, "sessions": len(group), "counts": dict(c),
                        "first_attempt_outcomes": dict(Counter(r["first_attempt_outcome"] for r in group)),
                        "raw_tokens_m": sum(r["raw_tokens_m"] for r in group),
                        "initial_attempt_tokens_m": sum(values), "initial_usage_sessions": len(values),
                        "initial_usage_complete": len(values) == len(group),
                        "core_tokens_m": None})
    return {"protocol": protocol, "scope": arm or "all_main_arms", "summary": summary, "sessions": rows,
        "source": str(base), "source_sessions": len(sessions),
        "source_sha256": hashlib.sha256((base / "sessions.json").read_bytes()).hexdigest(),
        "status": {"core_call_ledger": "recomputed", "core_token_usage": "not_recoverable_from_local_archive",
                   "within_attempt_rework": "unidentified", "supervised_first_attempt_outcomes": "unknown"},
        "excluded_sessions": [{"root": s["root"], "name": s["name"],
            "tokens_m": s["tokens_m"], "reason": "outside_final_main_design_or_environment_fault"}
            for s in sessions if s not in selected and (arm is None or s["arm"] == arm)
            and (s["agent"], s["model"]) == ("codex", "gpt-6-luna")],
        "limitations": ["No result-dependent exclusion of failed core calls or unsuccessful searches.",
            "Reviewers are separate from measured task streams and are not added to core cost.",
            "Historical rework inside an attempt cannot be removed without an evidence-backed interval ledger.",
            "Initial attempt usage is a partial sum when initial_usage_complete is false.",
            "No exact token subtraction from character shares or aggregate stage totals.",
            "core-v2 retains blocking run waits as execution; standalone zero-timeout probes remain monitoring. core-v1 is available for historical reproduction."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--protocol", choices=("core-v1", "core-v2"), default="core-v2")
    args = parser.parse_args()
    report = build_report(args.archive, args.arm, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["kind", "arm", "sessions", "core_calls", "excluded_calls", "raw_tokens_m",
                  "initial_attempt_tokens_m", "initial_usage_sessions", "initial_usage_complete", "core_tokens_m"]
        with args.csv.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for group in report["summary"]:
                row = {k: group.get(k) for k in fields}
                row["core_calls"] = group["counts"].get("core_calls", 0)
                row["excluded_calls"] = sum(group["counts"].get(k, 0) for k in
                    ("supervised_attempt", "monitoring", "stage_marker", "repeated_healthy_log_probe"))
                writer.writerow(row)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

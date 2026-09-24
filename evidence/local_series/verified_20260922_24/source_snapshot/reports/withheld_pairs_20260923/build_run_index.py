"""Create the complete 120-run accounting table without imputing missing work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "artifacts/withheld_pairs_20260923"
OUT = HERE / "data/run_index.csv"
JOB_PATTERN = re.compile(r"verifier_withheld_(train|eval)_20260923_k(\d+)s(\d+)_(\d+)_(random|withheld)\Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def load_audited_sets(report: Path, freeze_sha256: str) -> dict[tuple[int, int], dict]:
    accepted = {}
    for summary_path in sorted((report / "data").glob("k*_s*/analysis_summary.json")):
        match = re.fullmatch(r"k(\d+)_s(\d+)", summary_path.parent.name)
        if match is None:
            raise ValueError(f"unexpected set data directory: {summary_path.parent}")
        k, subset = map(int, match.groups())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("status") != "VALID" or summary.get("k") != k or
                summary.get("subset") != subset or summary.get("freeze_sha256") != freeze_sha256 or
                summary.get("completed_trainings") != 6 or summary.get("completed_evaluations") != 6):
            raise ValueError(f"invalid independent analysis: {summary_path}")
        archive = summary_path.parent / f"k{k}_s{subset}_audit.zip"
        sidecar = archive.with_suffix(".zip.sha256")
        if not archive.is_file() or not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != (
                f"{digest(archive)}  {archive.name}\n"):
            raise ValueError(f"archive SHA missing or mismatched: {archive}")
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if (len(names) != len(set(names)) or handle.testzip() is not None or
                    any(name.startswith("/") or "\\" in name or ".." in Path(name).parts or
                        name.endswith(".safetensors") for name in names)):
                raise ValueError(f"invalid audit ZIP: {archive}")
            manifest = json.loads(handle.read("AUDIT_MANIFEST.json"))
            expected_names = {item["path"] for item in manifest} | {
                "AUDIT_MANIFEST.json", "CHECKPOINT_INVENTORY.json"}
            if set(names) != expected_names:
                raise ValueError(f"audit ZIP manifest differs: {archive}")
            for item in manifest:
                payload = handle.read(item["path"])
                if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest().upper() != item["sha256"]:
                    raise ValueError(f"audit ZIP member hash differs: {item['path']}")
            member = summary_path.relative_to(report.parents[1]).as_posix()
            if handle.read(member) != summary_path.read_bytes():
                raise ValueError(f"audit summary differs from ZIP: {summary_path}")
            inventory = json.loads(handle.read("CHECKPOINT_INVENTORY.json"))
            if len(inventory) != 6 or any(item.get("weights_in_audit_zip") is not False for item in inventory):
                raise ValueError(f"checkpoint inventory incomplete: {archive}")
        accepted[(k, subset)] = summary
    return accepted


def _status(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("status")
    if value not in ("STARTED", "DONE", "FAILED"):
        raise ValueError(f"invalid run status: {path}")
    return value


def remote_job_states(freeze: dict, events: Iterable[dict]) -> dict[tuple[int, int, int, str, str], str]:
    allowed = {(entry["k"], entry["subset"], seed, arm)
               for entry in freeze["entries"] if entry["status"] == "FEASIBLE"
               for seed in freeze["seeds"] for arm in ("random", "withheld")}
    states = {}
    for event in events:
        if "job" not in event:
            continue
        match = JOB_PATTERN.fullmatch(event["job"])
        if match is None:
            raise ValueError(f"unknown remote job: {event['job']}")
        stage, k, subset, seed, arm = match.groups()
        row_key = (int(k), int(subset), int(seed), arm)
        if row_key not in allowed:
            raise ValueError(f"remote job outside frozen design: {event['job']}")
        status = event.get("status")
        if status not in ("STARTED", "DONE", "FAILED", "SKIPPED_ACCEPTED"):
            raise ValueError(f"invalid remote job status: {event['job']} {status}")
        states[(*row_key, stage)] = status
    return states


def load_queue_logs(paths: Iterable[Path]) -> list[dict]:
    events = []
    sources = {}
    for path in paths:
        file_sets = set()
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError(f"queue event is not an object: {path}:{number}")
                    match = JOB_PATTERN.fullmatch(str(event.get("job", "")))
                    if match is not None:
                        file_sets.add((int(match.group(2)), int(match.group(3))))
                    events.append(event)
        for set_key in file_sets:
            if set_key in sources:
                raise ValueError(f"duplicate remote set k{set_key[0]} s{set_key[1]}: "
                                 f"{sources[set_key]} and {path}")
            sources[set_key] = path
    return events


def build_index(freeze: dict, root: Path,
                audited_sets: dict[tuple[int, int], dict] | None = None,
                remote_events: Iterable[dict] | None = None) -> list[dict]:
    audited_sets = audited_sets or {}
    remote_states = remote_job_states(freeze, remote_events or ())
    rows = []
    for entry in freeze["entries"]:
        k, subset = entry["k"], entry["subset"]
        audit = audited_sets.get((k, subset))
        if audit is not None and (entry["status"] != "FEASIBLE" or audit.get("status") != "VALID" or
                                  audit.get("k") != k or audit.get("subset") != subset or
                                  audit.get("completed_trainings") != 6 or
                                  audit.get("completed_evaluations") != 6):
            raise ValueError(f"invalid audited set: k{k} s{subset}")
        for seed in freeze["seeds"]:
            for arm in ("random", "withheld"):
                prefix = f"k{k}_s{subset}_seed{seed}_{arm}"
                if entry["status"] != "FEASIBLE":
                    statuses = ["DESIGN_INFEASIBLE"] * 4
                else:
                    adapter_receipt = root / "adapters" / prefix / "training_receipt.json"
                    training = _status(root / "runs/training" / f"adapters__{prefix}.status.json")
                    if training == "DONE" and not adapter_receipt.is_file():
                        raise ValueError(f"DONE training lacks adapter receipt: {prefix}")
                    if training is None and list((root / "failures" / f"{prefix}_train").glob("*.json")):
                        training = "FAILED"
                    statuses = [training or "NOT_STARTED"]
                    for split in ("withheld", "control"):
                        statuses.append(_status(root / "rankings/status" / f"{prefix}_{split}.json") or "NOT_STARTED")
                    statuses.append(_status(root / "rankings/atomic" / f"{prefix}_atomic.json") or "NOT_STARTED")
                    if audit is not None:
                        if any(status not in ("DONE", "NOT_STARTED") for status in statuses):
                            raise ValueError(f"local raw status contradicts audited set: {prefix}")
                        statuses = ["DONE"] * 4
                rows.append({"k": k, "subset": subset, "seed": seed, "arm": arm,
                             "training_status": statuses[0],
                             "withheld_evaluation_status": statuses[1],
                             "control_evaluation_status": statuses[2],
                             "atomic_evaluation_status": statuses[3],
                             "remote_training_status": remote_states.get((k, subset, seed, arm, "train"), "UNKNOWN"),
                             "remote_evaluation_status": remote_states.get((k, subset, seed, arm, "eval"), "UNKNOWN"),
                             "design_reason": entry.get("reason", "")})
    if len(rows) != 120:
        raise ValueError("frozen design is not 120 runs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-log", type=Path, action="append", default=[],
                        help="read-only copied remote queue JSONL snapshot; repeat for each active set")
    parser.add_argument("--out", type=Path, default=OUT, help="destination CSV path")
    args = parser.parse_args()
    freeze = json.loads((ROOT / "FREEZE.json").read_text(encoding="utf-8"))
    rows = build_index(freeze, ROOT, load_audited_sets(HERE, digest(ROOT / "FREEZE.json")),
                       remote_events=load_queue_logs(args.queue_log))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete_evaluations = sum(all(row[key] == "DONE" for key in
                                   ("withheld_evaluation_status", "control_evaluation_status",
                                    "atomic_evaluation_status")) for row in rows)
    print(json.dumps({"planned_trainings": len(rows),
                      "design_infeasible": sum(row["training_status"] == "DESIGN_INFEASIBLE" for row in rows),
                      "completed_trainings": sum(row["training_status"] == "DONE" for row in rows),
                      "complete_evaluation_sets": complete_evaluations,
                      "remote_training_done": sum(row["remote_training_status"] in
                                                  ("DONE", "SKIPPED_ACCEPTED") for row in rows),
                      "remote_evaluation_done": sum(row["remote_evaluation_status"] in
                                                    ("DONE", "SKIPPED_ACCEPTED") for row in rows)}, sort_keys=True))


if __name__ == "__main__":
    main()

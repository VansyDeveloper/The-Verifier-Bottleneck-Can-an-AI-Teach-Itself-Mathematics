from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "manifests"
ARCHIVES = ROOT / "archives"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def validate_generations(name: str, gate: dict):
    rows = read_jsonl(RUNS / "generations" / f"{name}.jsonl")
    assert len(rows) == 1000
    assert len({r["task_id"] for r in rows}) == 1000
    counts = Counter(r["operation"] for r in rows)
    assert counts == Counter({op: 200 for op in ("SH1", "SC2", "REV", "AC1", "AX1")})
    assert all(r["schema"] == "stage4.atomic-generation.v1" for r in rows)
    recomputed = {op: sum(r["correct"] for r in rows if r["operation"] == op) / counts[op] for op in counts}
    assert recomputed == gate["apply"]["by_operation"]
    assert sum(r["correct"] for r in rows) / len(rows) == gate["apply"]["overall"]
    return {"rows": len(rows), "unique_task_ids": 1000, "operation_counts": dict(counts), "recomputed_apply": recomputed}


def validate_data_manifest():
    manifest = load(MANIFESTS / "data_manifest.json")
    checked = {}
    for name, item in manifest.items():
        path = ROOT / "data" / name
        rows = read_jsonl(path)
        assert len(rows) == item["rows"]
        assert sha256(path) == item["sha256"].lower()
        checked[f"data/{name}"] = {"rows": len(rows), "sha256": item["sha256"]}
    return checked


def make_report(initial: dict, hard: dict, receipts: dict):
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "atomic_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "metric", "SH1", "SC2", "REV", "AC1", "AX1", "overall", "pass"])
        for name, gate in (("initial", initial), ("hard_cycle", hard)):
            for metric in ("plan", "apply"):
                by = gate[metric]["by_operation"]
                writer.writerow([name, metric, *(by[o] for o in ("SH1", "SC2", "REV", "AC1", "AX1")), gate[metric]["overall"], gate["pass"]])

    total_train = sum(r["wall_seconds"] for r in receipts.values())
    total_eval = initial["wall_seconds"] + hard["wall_seconds"]
    text = f"""# Stage 4 v2 atomic rehabilitation — final report

## Verdict

**FAILED_ATOMIC_GATE. Composition training and final composition tests were not opened.**

The preregistered initial run passed PLAN but failed APPLY because SH1 reached only {initial['apply']['by_operation']['SH1']:.1%}. The sole permitted hard-operation cycle raised SH1 to {hard['apply']['by_operation']['SH1']:.1%}, still below the 90% gate. SC2, REV, AC1 and AX1 APPLY remained at 100% in both evaluations.

This is an honest negative atomic pilot, not evidence for or against composition distillation. No composition comparison, confidence interval, or p-value exists because the prerequisite experiment was correctly stopped before composition training.

## Frozen gate results

| Run | PLAN | APPLY | SH1 | SC2 | REV | AC1 | AX1 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Initial | {initial['plan']['overall']:.1%} | {initial['apply']['overall']:.1%} | {initial['apply']['by_operation']['SH1']:.1%} | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |
| Hard cycle | {hard['plan']['overall']:.1%} | {hard['apply']['overall']:.1%} | {hard['apply']['by_operation']['SH1']:.1%} | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |

Final PLAN by operation was SH1 {hard['plan']['by_operation']['SH1']:.1%}, SC2 {hard['plan']['by_operation']['SC2']:.1%}, REV {hard['plan']['by_operation']['REV']:.1%}, AC1 {hard['plan']['by_operation']['AC1']:.1%}, AX1 {hard['plan']['by_operation']['AX1']:.1%}. Overall PLAN {hard['plan']['overall']:.1%} passed the registered 95% threshold, but APPLY-each and spread did not.

## Actual budget

| Phase | Examples | Epochs | Optimizer steps | Loss-bearing tokens | LR | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Initial scaffold | {receipts['phase1']['examples']:,} | {receipts['phase1']['epochs']} | {receipts['phase1']['optimizer_steps']} | {receipts['phase1']['loss_bearing_target_tokens']:,} | {receipts['phase1']['lr']:.1e} | {receipts['phase1']['wall_seconds']:.1f} s |
| Initial plain + PLAN replay | {receipts['phase2']['examples']:,} | {receipts['phase2']['epochs']} | {receipts['phase2']['optimizer_steps']} | {receipts['phase2']['loss_bearing_target_tokens']:,} | {receipts['phase2']['lr']:.1e} | {receipts['phase2']['wall_seconds']:.1f} s |
| Sole hard cycle | {receipts['hard']['examples']:,} | {receipts['hard']['epochs']} | {receipts['hard']['optimizer_steps']} | {receipts['hard']['loss_bearing_target_tokens']:,} | {receipts['hard']['lr']:.1e} | {receipts['hard']['wall_seconds']:.1f} s |

Recorded training time was {total_train:.1f} s; successful frozen-dev evaluation time was {total_eval:.1f} s. Runtime microbatch was 8 with accumulation 8 (effective batch 64). Two preflight attempts were retained as FAILED: microbatch 4 for poor throughput and microbatch 16 for WDDM saturation. They produced no checkpoint and did not consume a scientific dev cycle.

## Integrity and stopping

- Atomic train/dev/hard datasets contain 25,000 / 1,000 / 20,000 rows and are hash-bound.
- Each gate is backed by 1,000 raw generations: 200 per operation and 1,000 unique task IDs.
- Reported APPLY accuracies were recomputed from raw generations during release validation.
- The single registered SH1/SC2 hard cycle was used; no additional tuning was performed.
- `composition_unlocked=false`; discover/distill data, composition adapters and final tests were not created or accessed in v2.

## Interpretation and next hypothesis

The failure is isolated to exact SH1 state execution, not operation selection: final SH1 PLAN was {hard['plan']['by_operation']['SH1']:.1%}, while SH1 APPLY was {hard['apply']['by_operation']['SH1']:.1%}. A future, separately preregistered v3 should change SH1 supervision or representation rather than silently extending this pilot. It must start from a new isolated directory and cannot reinterpret this v2 run as successful.
"""
    (REPORTS / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
    (ROOT / "STATUS.md").write_text("# Status\n\nFAILED_ATOMIC_GATE. Composition was not unlocked. See `reports/FINAL_REPORT.md`.\n", encoding="utf-8")


def archive(name: str, predicate):
    path = ARCHIVES / name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_file() and "archives" not in rel.parts and "__pycache__" not in rel.parts and predicate(rel):
                zf.write(p, Path("stage4_distill_v2") / rel)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    initial = load(RUNS / "atomic_initial_gate.json")
    hard = load(RUNS / "atomic_hard_cycle_gate.json")
    decision = load(RUNS / "ATOMIC_DECISION.json")
    assert not initial["pass"] and not hard["pass"]
    assert decision["status"] == "FAILED" and decision["composition_unlocked"] is False
    receipts = {
        "phase1": load(RUNS / "initial_phase1_training.json"),
        "phase2": load(RUNS / "initial_phase2_training.json"),
        "hard": load(RUNS / "hard_cycle_training.json"),
    }
    validation = {
        "status": "PASS",
        "decision": "FAILED_ATOMIC_GATE",
        "data": validate_data_manifest(),
        "generations": {
            "atomic_initial": validate_generations("atomic_initial", initial),
            "atomic_hard_cycle": validate_generations("atomic_hard_cycle", hard),
        },
        "json_files_validated": 0,
        "jsonl_files_validated": 0,
    }
    for p in ROOT.rglob("*.json"):
        if "archives" not in p.parts:
            load(p); validation["json_files_validated"] += 1
    for p in ROOT.rglob("*.jsonl"):
        if "archives" not in p.parts:
            read_jsonl(p); validation["jsonl_files_validated"] += 1
    dump(MANIFESTS / "VALIDATION.json", validation)
    dump(MANIFESTS / "RUN_INDEX.json", {
        "schema": "stage4.v2.run-index.v1",
        "terminal_status": "FAILED_ATOMIC_GATE",
        "runs": [
            {"id": "initial_preflight_micro4", "status": "FAILED", "checkpoint": False},
            {"id": "initial_preflight_micro16", "status": "FAILED", "checkpoint": False},
            {"id": "atomic_initial", "training": "DONE", "gate": "FAILED", "raw_generations": 1000},
            {"id": "atomic_hard_cycle", "training": "DONE", "gate": "FAILED", "raw_generations": 1000},
        ],
        "composition_unlocked": False,
    })
    make_report(initial, hard, receipts)
    files = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if (p.is_file() and "archives" not in rel.parts and "__pycache__" not in rel.parts
                and rel != Path("manifests/SHA256SUMS.json")):
            files[str(rel).replace("\\", "/")] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    dump(MANIFESTS / "SHA256SUMS.json", {"schema": "stage4.v2.sha256.v1", "files": files})
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    compact_roots = {"reports", "manifests", "configs", "code", "tests", "runs"}
    compact_names = {"PREREGISTRATION.md", "STATUS.md", ".gitignore"}
    compact = archive("stage4_distill_v2_audit_report.zip", lambda r: r.parts[0] in compact_roots or str(r) in compact_names)
    complete = archive("stage4_distill_v2_complete.zip", lambda r: True)
    dump(ARCHIVES / "RELEASE.json", {"schema": "stage4.v2.release.v1", "status": "FAILED_ATOMIC_GATE", "archives": [compact, complete]})
    print(json.dumps({"validation": validation, "archives": [compact, complete]}, indent=2))


if __name__ == "__main__":
    main()

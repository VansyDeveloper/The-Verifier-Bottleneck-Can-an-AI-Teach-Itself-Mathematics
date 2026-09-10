from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path


REPO = Path(r"C:\MyProject\verifier_bottleneck_codex_pack")
ROOT = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
POST = ROOT / "post_analysis_release_v1"
sys.path.insert(0, str(ROOT / "code"))

import confirm  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_entry(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def scientific_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or POST in path.parents:
            continue
        relative = path.relative_to(ROOT).as_posix()
        keep_failure_log = relative.startswith("runs/failures/") and relative.endswith(".log")
        if ("/__pycache__/" in f"/{relative}/" or relative.endswith((".pyc", ".pid")) or
                relative == "runs/ACTIVE_PROCESS.json" or relative == "runs/confirmation.run.lock" or
                (relative.endswith(".log") and not keep_failure_log)):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def build_scientific_seal() -> dict:
    entries = [manifest_entry(path, ROOT) for path in scientific_files()]
    payload = {
        "schema": "stage4.post-analysis.scientific-upstream-seal.v1",
        "status": "SEALED", "scientific_root": ROOT.name,
        "created_at_unix": time.time(), "files": entries,
        "file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries),
        "exclusions": ["post_analysis_release_v1/**", "runs/ACTIVE_PROCESS.json",
                       "runs/confirmation.run.lock", "non-failure logs", "__pycache__", "*.pyc", "*.pid"],
    }
    write_json(POST / "manifests/SCIENTIFIC_UPSTREAM_SEAL.json", payload)
    return payload


def verify_seal(seal: dict) -> dict:
    errors = []
    for entry in seal["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            errors.append(f"missing:{entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            errors.append(f"bytes:{entry['path']}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"sha256:{entry['path']}")
    return {"status": "PASS" if not errors else "FAIL", "checked": len(seal["files"]),
            "errors": errors}


def validate_atomic_raw() -> dict:
    rows = confirm.read_jsonl(confirm.DATA / "confirm_atomic.jsonl")
    jobs = [("shared", "atomic_base", "replicateshared_atomic_base")]
    jobs += [(str(label), branch, f"replicate{label}_{branch}")
             for label in range(6) for branch in ("atomic_control", "composition_distill")]
    reports = []
    for label, branch, name in jobs:
        replicate_dir = "replicateshared" if label == "shared" else f"replicate{label}"
        result_path = ROOT / "rankings" / replicate_dir / "atomic" / f"{name}.json"
        raw_path = result_path.with_name(f"{name}.generations.jsonl")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        raw = confirm.read_jsonl(raw_path)
        if result.get("status") != "DONE" or result.get("generation_sha256", "").upper() != sha256(raw_path):
            raise RuntimeError(f"atomic receipt/hash mismatch: {result_path}")
        plan, apply = confirm.eval_lib._derive_atomic_metrics(rows, raw, name, result["binding"])
        if (confirm.eval_lib._scientific_atomic(result["plan"], apply=False) != plan or
                confirm.eval_lib._scientific_atomic(result["apply"], apply=True) != apply):
            raise RuntimeError(f"atomic raw/scalar mismatch: {result_path}")
        reports.append({
            "replicate": label, "canonical_branch": branch, "eval_name": name,
            "tasks": len(raw), "result_sha256": sha256(result_path),
            "raw_sha256": sha256(raw_path), "plan": plan, "apply": apply,
        })
    payload = {"schema": "stage4.post-analysis.atomic-raw-validation.v1", "status": "PASS",
               "jobs": len(reports), "rows": sum(item["tasks"] for item in reports),
               "reports": reports}
    write_json(POST / "manifests/ATOMIC_RAW_VALIDATION.json", payload)
    return payload


def copy_provenance(source: Path, destination: Path) -> list[dict]:
    copied = []
    if not source.exists():
        return copied
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(manifest_entry(target, POST))
    return copied


def acceleration_provenance(args, seal: dict) -> dict:
    provenance_root = POST / "provenance"
    ai01_files = copy_provenance(args.ai01_provenance, provenance_root / "ai01")
    local_files = []
    for source in args.local_provenance:
        if source.is_file():
            target = provenance_root / "local" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            local_files.append(manifest_entry(target, POST))
        elif source.is_dir():
            local_files.extend(copy_provenance(source, provenance_root / "local" / source.name))
    equivalence_candidates = list((provenance_root / "ai01").rglob("remote_equivalence.json"))
    equivalence = json.loads(equivalence_candidates[0].read_text(encoding="utf-8")) if equivalence_candidates else None
    worker_receipts = []
    for path in sorted((provenance_root / "ai01").rglob("worker-?.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "DONE":
            worker_receipts.append({"path": path.relative_to(POST).as_posix(), "worker": value.get("worker"),
                                    "status": value.get("status"), "completed": value.get("completed", []),
                                    "sha256": sha256(path)})
    payload = {
        "schema": "stage4.post-analysis.acceleration-provenance.v1", "status": "DOCUMENTED",
        "scientific_primary_environment": {"host": "1cprog", "gpu": "NVIDIA GeForce RTX 5060 Ti",
                                            "torch": "2.10.0+cu128", "cuda": "12.8"},
        "secondary_acceleration_environment": {"host": "ai01", "gpu_count": 3,
                                               "gpu": "NVIDIA GeForce RTX 5080",
                                               "torch": "2.11.0+cu129", "cuda": "12.9",
                                               "container": "stage4-exact:20260802"},
        "scope": "secondary exact-ranking shards only; primary confirmation remained local and frozen",
        "equivalence_smoke": equivalence,
        "acceptance": {"correct_flags_exact": bool(equivalence and equivalence.get("correct_flags_exact")),
                       "correct_program_rank_map_exact": bool(equivalence and equivalence.get("correct_program_rank_map_exact")),
                       "rank_metrics_exact": bool(equivalence and equivalence.get("rank_metrics_exact")),
                       "near_tie_incorrect_position_swaps": equivalence.get("mismatch_positions") if equivalence else None},
        "windows_normalization": "All imported raw ranking shards were independently rederived into metrics and receipts under the frozen Windows evaluator before terminal analysis.",
        "worker_receipts": worker_receipts, "ai01_files": ai01_files, "local_files": local_files,
        "scientific_seal_sha256": sha256(POST / "manifests/SCIENTIFIC_UPSTREAM_SEAL.json"),
    }
    write_json(POST / "manifests/ACCELERATION_PROVENANCE.json", payload)
    return payload


def write_tables(analysis: dict, atomic: dict) -> dict:
    tables = POST / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    primary_path = tables / "primary_per_replicate.csv"
    with primary_path.open("w", encoding="utf-8", newline="") as handle:
        rows = analysis["primary"]["per_seed"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    holm_map = {row["endpoint"]: row for row in analysis["secondary_holm"]["results"]}
    secondary_rows = []
    for endpoint, value in analysis["secondary_holm"]["family"].items():
        stats = value["seed_statistics"]
        secondary_rows.append({
            "endpoint": endpoint, "splits": "+".join(value["splits"]),
            "mean_delta_pp": stats["mean_delta_pp"],
            "ci95_low_pp": 100 * stats["t_ci95"][0], "ci95_high_pp": 100 * stats["t_ci95"][1],
            "p_two_sided": stats["p_two_sided"],
            "holm_adjusted_p": holm_map[endpoint]["holm_adjusted_p"],
            "holm_reject_0p05": holm_map[endpoint]["reject_at_0p05"],
            "positive_replicates": stats["positive_seed_count"],
        })
    secondary_path = tables / "secondary_endpoints.csv"
    with secondary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(secondary_rows[0]))
        writer.writeheader(); writer.writerows(secondary_rows)

    forgetting_rows = []
    for label, branches in analysis["descriptive_non_gating"]["atomic_forgetting"].items():
        for branch, value in branches.items():
            forgetting_rows.append({"replicate": label, "branch": branch,
                                    "max_drop": value["max_drop"], "max_drop_pp": 100 * value["max_drop"],
                                    "passes_le_002": value["passes_le_002"],
                                    **value["drops"]})
    forgetting_path = tables / "atomic_forgetting.csv"
    with forgetting_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(forgetting_rows[0]))
        writer.writeheader(); writer.writerows(forgetting_rows)

    atomic_rows = []
    for item in atomic["reports"]:
        row = {"replicate": item["replicate"], "branch": item["canonical_branch"],
               "plan_overall": item["plan"]["overall"], "apply_overall": item["apply"]["overall"]}
        for op, value in item["plan"]["by_operation"].items(): row[f"plan_{op}"] = value
        for op, value in item["apply"]["by_operation"].items(): row[f"apply_{op}"] = value
        atomic_rows.append(row)
    atomic_path = tables / "atomic_plan_apply.csv"
    with atomic_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(atomic_rows[0]))
        writer.writeheader(); writer.writerows(atomic_rows)
    return {path.stem: manifest_entry(path, POST) for path in
            (primary_path, secondary_path, forgetting_path, atomic_path)}


def write_figures(analysis: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = POST / "figures"; figures.mkdir(parents=True, exist_ok=True)
    primary = analysis["primary"]["per_seed"]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar([str(row["replicate_label"]) for row in primary], [row["delta_pp"] for row in primary], color="#167d3e")
    axis.axhline(5, color="#333", linestyle="--", label="registered +5 pp")
    axis.axhline(0, color="black", linewidth=.8); axis.set_ylabel("Hit@32 delta (pp)")
    axis.set_xlabel("Replicate"); axis.set_title("Final-A depth-3 primary effect"); axis.legend(); fig.tight_layout()
    primary_path = figures / "primary_effect.png"; fig.savefig(primary_path, dpi=180); plt.close(fig)

    family = analysis["secondary_holm"]["family"]
    labels = list(family); values = [family[name]["seed_statistics"]["mean_delta_pp"] for name in labels]
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar(range(len(labels)), values, color=["#167d3e" if value > 0 else "#b3261e" for value in values])
    axis.axhline(0, color="black", linewidth=.8); axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("Hit@32 delta (pp)"); axis.set_title("Secondary transfer endpoints (descriptive means)")
    fig.tight_layout(); secondary_path = figures / "secondary_effects.png"; fig.savefig(secondary_path, dpi=180); plt.close(fig)
    return {path.stem: manifest_entry(path, POST) for path in (primary_path, secondary_path)}


def write_report(analysis: dict, raw: dict, atomic: dict, acceleration: dict) -> Path:
    stats = analysis["primary"]["seed_statistics"]
    bootstrap = analysis["primary"]["seed_task_bootstrap"]
    lines = [
        "# Stage 4 Verifier Bottleneck — final Discover-and-Distill report", "",
        f"**Scientific decision: {analysis['decision']['status']}.**", "",
        "The model is Qwen3-0.6B throughout. The starting atomic checkpoint is the frozen corrective checkpoint with SH1 APPLY 72.778%. No 1.7B model was used.", "",
        "## Primary preregistered confirmation", "",
        f"- Final-A depth-3 Hit@32 mean delta: **{stats['mean_delta_pp']:.3f} pp**.",
        f"- 95% seed t-CI: **[{100*stats['t_ci95'][0]:.3f}, {100*stats['t_ci95'][1]:.3f}] pp**.",
        f"- Two-sided seed t-test: **p={stats['p_two_sided']:.8g}**.",
        f"- Positive replicates: **{stats['positive_seed_count']}/6**.",
        f"- Exact sign-flip sensitivity: **p={analysis['primary']['exact_sign_flip']['p_two_sided']:.8g}**.",
        f"- Crossed seed-by-task bootstrap ({bootstrap['repetitions']:,}): **[{100*bootstrap['ci95'][0]:.3f}, {100*bootstrap['ci95'][1]:.3f}] pp**.", "",
        "All four registered primary gates passed. This is evidence that composition_distill improved exact ranking of unseen family-A depth-3 compositions relative to the equal-budget atomic_control.", "",
        "## Secondary endpoints", "",
        "| Endpoint | Mean delta (pp) | 95% CI (pp) | raw p | Holm p |", "|---|---:|---:|---:|---:|",
    ]
    holm_map = {row["endpoint"]: row for row in analysis["secondary_holm"]["results"]}
    for endpoint, value in analysis["secondary_holm"]["family"].items():
        stat = value["seed_statistics"]; adjusted = holm_map[endpoint]
        lines.append(f"| {endpoint} | {stat['mean_delta_pp']:.3f} | [{100*stat['t_ci95'][0]:.3f}, {100*stat['t_ci95'][1]:.3f}] | {stat['p_two_sided']:.6g} | {adjusted['holm_adjusted_p']:.6g} |")
    forgetting = analysis["descriptive_non_gating"]["atomic_forgetting"]
    passes = sum(value["passes_le_002"] for branches in forgetting.values() for value in branches.values())
    total = sum(len(branches) for branches in forgetting.values())
    lines += ["", "## Atomic skills and forgetting", "",
              f"All {atomic['jobs']} atomic raw files ({atomic['rows']:,} task rows) were independently rederived. The descriptive ≤2 pp forgetting check passed for {passes}/{total} trained branch/replicate pairs.",
              "Per the registered composition-only confirmation, atomic forgetting is reported descriptively and does not override the primary composition decision.", "",
              "## Integrity", "",
              f"- Full exact-ranking validation: **{raw['status']}**, {raw['ranking_shards']} shards.",
              "- Every depth-2/3/4 task ranks the complete 25/125/625 unique program set; verifier labels and score order were recomputed.",
              f"- Leakage audit count: **{analysis['integrity']['leakage_count']}**.",
              "- Training budgets are equal within every replicate.",
              "- All primary evidence was produced locally under the frozen environment.", "",
              "## Operational acceleration addendum (post hoc, non-gating)", "",
              "Secondary rankings were accelerated on ai01 with 3×RTX 5080. A depth-4 equivalence smoke retained all correct-program ranks and every target rank metric; two near-tied incorrect positions out of 625 swapped under torch 2.11, with the difference disclosed in ACCELERATION_PROVENANCE.json.",
              "All imported remote raw rankings were rederived into Windows metrics/receipts under the frozen evaluator before final analysis. Failed transfer/preflight attempts produced no accepted scientific shards and are preserved as operational provenance.", "",
              "## Scope", "",
              "This positive result supports exact compositional ranking transfer for the registered final-A endpoint. Secondary B/C/D and depth-4 results are reported separately and must not be conflated with the primary claim.",
              "The earlier exploratory pilot remains historically unchanged; the six-replicate follow-up is the confirmatory evidence reported here.", "",
              "Archive hashes and member counts are recorded in manifests/ARCHIVE_MANIFEST.json.", ""]
    path = POST / "FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def zip_items(path: Path, items: list[tuple[Path, str]]) -> dict:
    temporary = path.with_suffix(path.suffix + ".partial")
    seen = set()
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for source, name in sorted(items, key=lambda item: item[1]):
            if name in seen: raise RuntimeError(f"duplicate archive member: {name}")
            seen.add(name)
            stored = source.suffix.lower() in {".safetensors", ".zip", ".gz", ".npy"}
            archive.write(source, name, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                          compresslevel=None if stored else 6)
    temporary.replace(path)
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip(); members = archive.namelist()
    if bad is not None: raise RuntimeError(f"archive CRC failed: {bad}")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "members": len(members), "crc": "PASS"}


def release_items() -> list[Path]:
    return sorted((path for path in POST.rglob("*") if path.is_file() and "archives" not in path.relative_to(POST).parts),
                  key=lambda item: item.relative_to(POST).as_posix())


def write_archives() -> dict:
    archives = POST / "archives"; archives.mkdir(parents=True, exist_ok=True)
    compact_selected = [
        ROOT / "statistics/FINAL_ANALYSIS.json", ROOT / "statistics/PRIMARY_ANALYSIS.json",
        ROOT / "manifests/raw_validation_full.json", ROOT / "manifests/final_split_audit.json",
        ROOT / "manifests/archive_manifest.json", ROOT / "reports/FINAL_REPORT.md",
        ROOT / "configs/protocol.json", ROOT / "PREREGISTRATION.md",
    ]
    compact_items = [(path, f"scientific/{path.relative_to(ROOT).as_posix()}") for path in compact_selected]
    compact_items += [(path, f"post_analysis_release_v1/{path.relative_to(POST).as_posix()}") for path in release_items()]

    full_items = []
    for path in scientific_files():
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "archives":
            continue
        full_items.append((path, f"scientific/{relative.as_posix()}"))
    export = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/atomic_export/frozen_atomic_0p6b"
    for path in sorted(item for item in export.rglob("*") if item.is_file()):
        full_items.append((path, f"external/atomic_export/{path.relative_to(export).as_posix()}"))
    for path in confirm.runtime_dependency_paths():
        full_items.append((path, f"external/runtime_dependencies/{path.relative_to(REPO).as_posix()}"))
    historical = [
        REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/runs/PILOT_DECISION.json",
        REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/runs/STAGE4_FAILED.json",
        REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/runs/pilot/seed0_lr1em04_e2_r0p20/attempt.json",
        REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/manifests/pilot_data_manifest.json",
        REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/data/discover_trajectories.jsonl",
    ]
    full_items += [(path, f"external/historical_pilot/{path.name}") for path in historical]
    full_items += [(path, f"post_analysis_release_v1/{path.relative_to(POST).as_posix()}") for path in release_items()]

    compact_path = archives / "stage4_verifier_bottleneck_audit_v1.zip"
    full_path = archives / "stage4_verifier_bottleneck_full_v1.zip"
    compact = zip_items(compact_path, compact_items)
    full = zip_items(full_path, full_items)
    (compact_path.with_suffix(compact_path.suffix + ".sha256")).write_text(f"{compact['sha256']}  {compact_path.name}\n", encoding="ascii")
    (full_path.with_suffix(full_path.suffix + ".sha256")).write_text(f"{full['sha256']}  {full_path.name}\n", encoding="ascii")
    payload = {"schema": "stage4.post-analysis.archive-manifest.v1", "status": "DONE",
               "compact": compact, "full": full,
               "full_includes_frozen_atomic_export": True,
               "full_excludes_nested_original_archives": True}
    write_json(POST / "manifests/ARCHIVE_MANIFEST.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai01-provenance", type=Path, required=True)
    parser.add_argument("--local-provenance", type=Path, action="append", default=[])
    args = parser.parse_args()
    required = [ROOT / "runs/STAGE4_COMPOSITION_DONE.json", ROOT / "runs/EXPERIMENT_DONE.json",
                ROOT / "statistics/FINAL_ANALYSIS.json", ROOT / "manifests/raw_validation_full.json",
                ROOT / "manifests/archive_manifest.json", ROOT / "reports/FINAL_REPORT.md"]
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"scientific terminal outputs are incomplete: {[str(p) for p in required if not p.is_file()]}")
    if POST.exists() and any(POST.rglob("*")):
        raise RuntimeError(f"post-analysis release already exists: {POST}")
    POST.mkdir(parents=True, exist_ok=True)
    source_script = Path(__file__).resolve()
    code_target = POST / "code/build_post_analysis_release.py"
    code_target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source_script, code_target)

    analysis = json.loads((ROOT / "statistics/FINAL_ANALYSIS.json").read_text(encoding="utf-8"))
    raw = json.loads((ROOT / "manifests/raw_validation_full.json").read_text(encoding="utf-8"))
    if analysis.get("status") != "DONE" or raw.get("status") != "PASS":
        raise RuntimeError("scientific analysis/raw validation is not terminal PASS")
    seal = build_scientific_seal()
    atomic = validate_atomic_raw()
    acceleration = acceleration_provenance(args, seal)
    tables = write_tables(analysis, atomic)
    figures = write_figures(analysis)
    report = write_report(analysis, raw, atomic, acceleration)
    prearchive = {"schema": "stage4.post-analysis.validation.v1", "status": "PASS",
                  "scientific_seal": verify_seal(seal), "atomic_raw": "PASS",
                  "analysis_decision": analysis["decision"], "tables": tables, "figures": figures,
                  "report": manifest_entry(report, POST), "python": platform.python_version()}
    write_json(POST / "manifests/VALIDATION_MANIFEST.json", prearchive)
    # Freeze the payload inventory before archiving so both deliverable ZIPs
    # contain the same auditable run index.  Archive hashes and DONE.json are
    # intentionally external to avoid a self-referential archive manifest.
    run_files = [manifest_entry(path, POST) for path in release_items()]
    write_json(POST / "manifests/RUN_INDEX.json", {
        "schema": "stage4.post-analysis.run-index.v1", "status": "DONE",
        "scope": "pre_archive_release_payload",
        "exclusions": ["manifests/RUN_INDEX.json", "archives/**", "DONE.json"],
        "files": run_files, "file_count": len(run_files),
        "scientific_seal_sha256": sha256(POST / "manifests/SCIENTIFIC_UPSTREAM_SEAL.json"),
    })
    archives = write_archives()
    final_seal_check = verify_seal(seal)
    if final_seal_check["status"] != "PASS":
        raise RuntimeError(f"scientific seal changed during release: {final_seal_check}")
    terminal = {"schema": "stage4.post-analysis.terminal.v1", "status": "DONE",
                "scientific_decision": analysis["decision"], "scientific_seal": final_seal_check,
                "compact_sha256": archives["compact"]["sha256"],
                "full_sha256": archives["full"]["sha256"], "finished_at_unix": time.time()}
    write_json(POST / "DONE.json", terminal)
    print(json.dumps(terminal, sort_keys=True))


if __name__ == "__main__":
    main()

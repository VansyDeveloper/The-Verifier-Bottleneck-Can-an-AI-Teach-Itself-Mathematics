"""Freeze preselected training and held-out data before model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from design import build_all_selections
from holdout import classify_programs, generate_conditioned


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8"
CONFIRM = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
PREVIOUS = REPO / "artifacts/trajectory_diversity_20260922"
SOURCE = LEGACY / "data/discover_trajectories.jsonl"
PAIR_SETS = REPO / "reports/trajectory_diversity_20260922/WITHHELD_PAIR_SETS.json"
OUTPUT = REPO / "artifacts/withheld_pairs_20260923"
sys.path.insert(0, str(LEGACY / "code"))
sys.path.insert(0, str(REPO / "reports/trajectory_diversity_20260922"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_exclusive_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode("utf-8") + b"\n" for row in rows)


def reserved_registry(core):
    import run_experiment as previous_run

    atomic = previous_run.atomic_reference(core)
    registry = core.FingerprintRegistry.from_atomic_reference(atomic)
    prior = {name: read_jsonl(LEGACY / "data" / f"{name}.jsonl")
             for name in ("train", "dev_a", "dev_b")}
    prior["trajectory_source"] = read_jsonl(SOURCE)
    for path in sorted((CONFIRM / "data").glob("final_*.jsonl")):
        prior[path.stem] = read_jsonl(path)
    prior["trajectory_diversity_holdout"] = read_jsonl(PREVIOUS / "data/holdout_final_a.jsonl")
    core._reserve_existing_rows(registry, prior)
    return registry


def prepare_all(output: Path = OUTPUT) -> dict:
    import composition_core as core

    source = read_jsonl(SOURCE)
    design = json.loads(PAIR_SETS.read_text(encoding="utf-8"))
    previous_freeze = json.loads((PREVIOUS / "SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    if sha256(SOURCE) != design["source_sha256"] or sha256(SOURCE) != previous_freeze["source_sha256"]:
        raise RuntimeError("source trajectory SHA-256 mismatch")
    if len(source) != 2250 or len(previous_freeze["target_lengths"]) != 2250:
        raise RuntimeError("source or target-length count mismatch")
    entries = build_all_selections(source, previous_freeze["target_lengths"], design)
    if len(entries) != 20 or (entries[0]["k"], entries[0]["subset"]) != (1, 1):
        raise RuntimeError("preselected set order changed")
    registry = reserved_registry(core)
    prior_tasks = len(registry.tasks)
    prior_states = len(registry.states)
    for entry in entries:
        if entry["status"] != "FEASIBLE":
            continue
        excluded = {tuple(pair) for pair in entry["pairs"]}
        basename = f"k{entry['k']}_s{entry['subset']}"
        files = {}
        for kind, offset in (("withheld", 0), ("control", 1)):
            seed = 86000 + entry["k"] * 100 + entry["subset"] * 2 + offset
            rows = generate_conditioned(core, registry, excluded, kind=kind, count=1000, seed=seed)
            target = output / "holdouts" / f"{basename}_{kind}.jsonl"
            payload = jsonl_bytes(rows)
            write_exclusive_or_verify(target, payload)
            files[kind] = {"path": target.relative_to(output).as_posix(), "sha256": sha256(target),
                           "rows": len(rows), "seed": seed, "task_ids_sha256": hashlib.sha256(
                               json_bytes(sorted(row["task_id"] for row in rows))).hexdigest().upper()}
            if kind == "withheld":
                classes = []
                for row in rows:
                    search = core.exact_shortest_solutions(row["start"], row["target"], row["p"], 3, limit=125)
                    classes.append(classify_programs((solution.program for solution in search.solutions), excluded))
                files[kind]["all_correct_require_pair"] = classes.count("withheld")
                files[kind]["alternative_correct_without_pair"] = classes.count("mixed")
        entry["holdouts"] = files
    code_paths = [HERE / name for name in ("design.py", "holdout.py", "prepare.py",
                                           "fp16_training.py", "run_first.py", "verify.py", "PROTOCOL.md")]
    code_paths += [LEGACY / "code" / name for name in ("composition_core.py", "composition_model.py", "composition_eval.py")]
    prior_config = json.loads((CONFIRM / "configs/protocol.json").read_text(encoding="utf-8"))
    training_config = {**prior_config["training"], "dtype": "float16"}
    manifest = {"schema": "verifier-bottleneck.withheld-pairs-freeze.v1", "status": "FROZEN",
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "source_sha256": sha256(SOURCE), "pair_sets_sha256": sha256(PAIR_SETS),
                "previous_target_lengths_sha256": sha256(PREVIOUS / "SELECTION_FROZEN.json"),
                "atomic_export_payload_sha256": previous_freeze["atomic_export_tree_sha256"],
                "atomic_pool_sha256": {name: sha256(REPO / "artifacts/stage4_sh1_v5_0p6b/data" / f"{name}.jsonl")
                                       for name in ("full_sh1", "control_apply")},
                "atomic_evaluation_sha256": sha256(CONFIRM / "data/confirm_atomic.jsonl"),
                "training_config": training_config,
                "code_sha256": {path.relative_to(REPO).as_posix(): sha256(path) for path in code_paths},
                "prior_task_fingerprints": prior_tasks, "prior_state_fingerprints": prior_states,
                "seeds": [85000, 85001, 85002], "entries": entries}
    write_exclusive_or_verify(output / "FREEZE.json", json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = prepare_all(args.output)
    print(json.dumps({"status": manifest["status"], "entries": len(manifest["entries"]),
                      "feasible": sum(entry["status"] == "FEASIBLE" for entry in manifest["entries"]),
                      "freeze_sha256": sha256(args.output / "FREEZE.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

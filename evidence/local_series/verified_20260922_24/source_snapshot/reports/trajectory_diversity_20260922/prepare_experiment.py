"""Freeze selections and code/input bindings before any new training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from selection import select_pair, validate_pair_budget


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8"
CONFIRM = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
sys.path.insert(0, str(LEGACY / "code"))
import composition_model as model_lib


SOURCE_SHA = "3C2BD26D16AA7BAA0EB2D6CC277C4FFD97C189B3C3405D6CD9FFEAE325C2550F"
EXPORT_TREE_SHA = "043BCFC701423F644982659731CE712A0CD36708ADC1981D51BD05DCAD2EFC22"
SEEDS = range(83000, 83006)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_export(export: Path, expected_payload_sha: str) -> str:
    actual = model_lib._export_payload_sha(export).upper()
    if actual != expected_payload_sha:
        raise RuntimeError("frozen atomic export payload SHA-256 changed")
    return actual


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/trajectory_diversity_20260922")
    args = parser.parse_args()
    source = LEGACY / "data/discover_trajectories.jsonl"
    export = LEGACY / "atomic_export/frozen_atomic_0p6b"
    if sha256(source) != SOURCE_SHA:
        raise RuntimeError("frozen trajectory pool SHA-256 changed")
    verify_export(export, EXPORT_TREE_SHA)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2250 or any(row.get("split") != "train" for row in rows):
        raise RuntimeError("unexpected trajectory source")
    tokenizer, _ = model_lib.tokenizer_and_ids(export)
    target_lengths = {}
    for row in rows:
        answer = model_lib.program_answer(row["program"]) + "\nTRACE: " + " -> ".join(
            model_lib.format_state(state) for state in row["states"][1:])
        target_lengths[row["trajectory_id"]] = len(tokenizer.encode(" " + answer + tokenizer.eos_token,
                                                                      add_special_tokens=False))
    selections = {}
    for seed in SEEDS:
        pair = select_pair(rows, target_lengths, seed=seed)
        budget = validate_pair_budget(rows, target_lengths, pair["random_ids"], pair["diverse_ids"])
        gap = pair["diverse"]["selection_objective"] - pair["random"]["selection_objective"]
        if gap < 0.03:
            raise RuntimeError(f"seed {seed} manipulation check failed: objective gap {gap:.6f} < 0.03")
        selections[str(seed)] = {"seed": seed, "random_ids": pair["random_ids"],
                                 "diverse_ids": pair["diverse_ids"], "budget": budget,
                                 "random_features": pair["random"], "diverse_features": pair["diverse"],
                                 "objective_gap": gap, "strata_quotas": pair["strata_quotas"]}
    config = json.loads((CONFIRM / "configs/protocol.json").read_text(encoding="utf-8"))
    paths = [HERE / "selection.py", HERE / "prepare_experiment.py", HERE / "EXPERIMENT_PROTOCOL.md",
             HERE / "run_experiment.py", HERE / "analyze_experiment.py",
             LEGACY / "code/composition_model.py", LEGACY / "code/composition_core.py",
             LEGACY / "code/composition_eval.py"]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise RuntimeError(f"cannot freeze absent code: {absent}")
    freeze = {"schema": "verifier-bottleneck.trajectory-selection-freeze.v1", "status": "FROZEN",
              "study_class": "exploratory-dataset-composition-intervention",
              "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "source_sha256": SOURCE_SHA, "atomic_export_tree_sha256": EXPORT_TREE_SHA,
              "source_confirm_protocol_sha256": sha256(CONFIRM / "configs/protocol.json"),
              "holdout_seed": 83100, "holdout_count": 1000, "holdout_family": "A", "holdout_depth": 3,
              "training_config": config["training"], "code_sha256": {str(path.relative_to(REPO)).replace("\\", "/"): sha256(path) for path in paths},
              "target_lengths": target_lengths, "selections": selections}
    path = args.output / "SELECTION_FROZEN.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != freeze:
            raise RuntimeError("existing selection freeze does not match current code or inputs")
    else:
        write_exclusive(path, freeze)
    print(json.dumps({"status": "FROZEN", "path": str(path), "sha256": sha256(path),
                      "seeds": list(SEEDS), "objective_gaps": {seed: selections[str(seed)]["objective_gap"] for seed in SEEDS}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

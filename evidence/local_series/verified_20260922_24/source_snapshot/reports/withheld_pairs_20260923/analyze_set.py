"""Audit one complete post-amendment withheld-pair set without pooling GPUs."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import analyze_first as audit


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing analysis differs: {path}")
        return
    path.write_bytes(data)


def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("empty analysis table")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def run(repo: Path, k: int, subset: int, out: Path) -> dict:
    audit.REPO = repo
    audit.ROOT = repo / "artifacts/withheld_pairs_20260923"
    audit.CONFIRM = repo / "artifacts/stage4_composition_confirm_v1_0p6b"
    audit.CORE_DIR = repo / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code"
    if audit.digest(audit.ROOT / "FREEZE.json") != audit.FREEZE_SHA256:
        raise ValueError("frozen manifest SHA-256 mismatch")
    freeze = json.loads((audit.ROOT / "FREEZE.json").read_text(encoding="utf-8"))
    matches = [row for row in freeze["entries"] if row["k"] == k and row["subset"] == subset]
    if len(matches) != 1 or matches[0]["status"] != "FEASIBLE":
        raise ValueError("requested set is absent or infeasible")
    entry = matches[0]
    allocation_path = audit.ROOT / "runs" / f"k{k}_s{subset}_allocation.json"
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    if (allocation.get("freeze_sha256") != audit.FREEZE_SHA256 or
            allocation.get("compute_dtype") != "torch.float16" or
            allocation.get("master_dtype") != "torch.float32"):
        raise ValueError("amended GPU allocation is invalid")

    base_atomic = json.loads((audit.CONFIRM / "rankings/replicateshared/atomic/replicateshared_atomic_base.json")
                             .read_text(encoding="utf-8"))
    if (base_atomic.get("status") != "DONE" or
            base_atomic.get("binding", {}).get("atomic_export_sha256") != freeze["atomic_export_payload_sha256"] or
            base_atomic.get("binding", {}).get("data_sha256") !=
            audit.digest(audit.CONFIRM / "data/confirm_atomic.jsonl")):
        raise ValueError("frozen atomic baseline is invalid")
    source = {row["trajectory_id"]: row for row in audit.read_jsonl(
        repo / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/data/discover_trajectories.jsonl")}
    rows: list[dict] = []
    curves: list[dict] = []
    atomic_rows: list[dict] = []
    selection_rows: list[dict] = []
    for seed in freeze["seeds"]:
        for arm in ("random", "withheld"):
            selection = entry["selections"][str(seed)]
            identities = selection[f"{arm}_ids"]
            if len(identities) != 750 or len(set(identities)) != 750:
                raise ValueError(f"selection incomplete: {seed}/{arm}")
            excluded = sum(audit.selected_pair_count(source, identities, tuple(pair))
                           for pair in entry["pairs"])
            selection_rows.append({
                "seed": seed, "arm": arm, "selected_trajectories": len(identities),
                "removed_trajectories": len(selection[f"{arm}_removed_ids"]),
                "selected_with_excluded_pairs": excluded,
                "composition_target_tokens_per_epoch": selection["budget"]["target_tokens_per_epoch_each"],
            })
            adapter_tree = audit.audit_training(seed, arm, audit.FREEZE_SHA256, k, subset)
            for split in ("withheld", "control"):
                result = audit.audit_ranking(seed, arm, split, entry["holdouts"][split],
                                             audit.FREEZE_SHA256, adapter_tree, k, subset)
                rows.append({"seed": seed, "arm": arm, "split": split,
                             "hit32": result["hit32"], "correct_mass": result["correct_mass"]})
                curves.extend({"seed": seed, "arm": arm, "split": split,
                               "k": rank, "hit_k": value}
                              for rank, value in enumerate(result["curve"], 1))
            atomic = audit.audit_atomic(seed, arm, audit.FREEZE_SHA256, adapter_tree, k, subset)
            for metric in ("plan", "apply"):
                for operation, accuracy in atomic[metric]["by_operation"].items():
                    atomic_rows.append({
                        "seed": seed, "arm": arm, "metric": metric, "operation": operation,
                        "accuracy": accuracy,
                        "drop_from_frozen_atomic":
                            base_atomic[metric]["by_operation"][operation] - accuracy,
                    })
    differences = []
    for seed in freeze["seeds"]:
        matched = {row["arm"]: row["hit32"] for row in rows
                   if row["seed"] == seed and row["split"] == "withheld"}
        differences.append(matched["withheld"] - matched["random"])
    result = {
        "schema": "withheld-pairs.amended-set-audited.v1", "status": "VALID",
        "freeze_sha256": audit.FREEZE_SHA256, "k": k, "subset": subset,
        "gpu_host": allocation["host"], "gpu_name": allocation["gpu_name"],
        "gpu_uuid": allocation["gpu_uuid"], "completed_trainings": 6,
        "completed_evaluations": 6,
        "withheld_hit32_difference": audit.paired_stats(differences),
        "per_seed_differences": dict(zip(freeze["seeds"], differences)),
        "interpretation": "one preselected set only; no inference over k or subsets",
    }
    write_once(out / "seed_metrics.csv", csv_bytes(rows))
    write_once(out / "selection_integrity.csv", csv_bytes(selection_rows))
    write_once(out / "hit_k.csv", csv_bytes(curves))
    write_once(out / "atomic_forgetting.csv", csv_bytes(atomic_rows))
    write_once(out / "analysis_summary.json",
               (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--subset", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.k, args.subset, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()

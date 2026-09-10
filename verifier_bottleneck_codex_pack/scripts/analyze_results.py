from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(path for path in args.runs.iterdir() if path.is_dir()):
        status = "DONE" if (run_dir / "DONE").exists() else "FAILED" if (run_dir / "FAILED").exists() else "INCOMPLETE"
        metrics = {}
        if (run_dir / "final_metrics.json").exists():
            metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        config = {}
        if (run_dir / "config.resolved.yaml").exists():
            config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8")) or {}
        row = {
            "run_id": run_dir.name,
            "status": status,
            "seed": manifest["seed"],
            "started_at": manifest["started_at"],
            "finished_at": manifest.get("finished_at"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "pass_at_1": metrics.get("pass_at_1"),
            "pass_at_4": metrics.get("pass_at_4"),
            "parse_rate": metrics.get("parse_rate"),
            "mean_reward": metrics.get("mean_reward"),
            "steps": metrics.get("steps"),
            "tasks": metrics.get("tasks"),
            "candidates": metrics.get("candidates"),
            "hit_at_1": metrics.get("hit_at_1"),
            "hit_at_8": metrics.get("hit_at_8"),
            "hit_at_32": metrics.get("hit_at_32"),
            "model_passes": metrics.get("model_passes"),
            "actual_completion_tokens": metrics.get("actual_completion_tokens"),
            "peak_cuda_bytes": metrics.get("peak_cuda_bytes"),
            "input": config.get("input"),
            "input_sha256": config.get("input_sha256"),
            "adapter": config.get("adapter"),
            "adapter_sha256": config.get("adapter_sha256"),
            "config_sha256": manifest.get("config_sha256"),
        }
        rows.append(row)
    output_csv = args.output / "run_index.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id", "status"])
        writer.writeheader()
        writer.writerows(rows)
    confirmatory_path = args.output / "three_seed_heldout_statistics.json"
    confirmatory = None
    if confirmatory_path.exists():
        confirmatory = json.loads(confirmatory_path.read_text(encoding="utf-8"))
    summary = {
        "runs": len(rows),
        "done": sum(row["status"] == "DONE" for row in rows),
        "failed": sum(row["status"] == "FAILED" for row in rows),
        "incomplete": sum(row["status"] == "INCOMPLETE" for row in rows),
        "scientific_ci_available": confirmatory is not None,
        "confirmatory_pilot": confirmatory["aggregate"] if confirmatory else None,
        "reason": (
            "Three-seed 150-step confirmatory pilot; this is not the 400-step full protocol."
            if confirmatory
            else "No confirmatory statistics file is available."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_index": str(output_csv), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

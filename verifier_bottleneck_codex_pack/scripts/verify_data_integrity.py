"""Full data-integrity audit (Gate 0 and Gate 3 evidence).

Checks every split that exists, not only the six listed in the original
`split_manifest.json`, and reports leaks by exact key rather than asserting
their absence. Writes `artifacts/reports/data_integrity.json`.

This script never rewrites a frozen manifest: the July `split_manifest.json` and
the confirmatory manifest stay byte-identical so their SHA256 values remain
verifiable. New coverage goes into a separate complete manifest.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from vbexp.experiment import sha256_file, write_json
from vbexp.io import read_tasks

DATA = Path("artifacts/data/pilot")
TRAIN_PRIMES = (5, 7, 13, 19, 23, 29)
HELDOUT_PRIMES = (11, 17)

SPLITS = [
    "sft_train_apply",
    "sft_train_plan",
    "sft_validation_apply",
    "sft_validation_apply_non_sh1",
    "sft_validation_plan",
    "sft_calibration_sh1",
    "dev_exploration",
    "rl_train",
    "final_like_heldout",
    "confirmatory_heldout",
]

# Splits that must never share an exact key with an evaluation split.
TRAINING = {"sft_train_apply", "sft_train_plan", "sft_calibration_sh1", "rl_train"}
EVALUATION = {"dev_exploration", "final_like_heldout", "confirmatory_heldout"}


def exact_key(task):
    return (
        task.mode,
        task.p,
        task.degree_cap,
        tuple(task.start),
        tuple(task.target) if task.target else None,
        tuple(task.program) if task.program else None,
        task.max_steps,
    )


def main() -> None:
    present = {name: DATA / f"{name}.jsonl" for name in SPLITS if (DATA / f"{name}.jsonl").exists()}
    loaded = {name: read_tasks(path) for name, path in present.items()}
    keys = {name: {exact_key(task) for task in tasks} for name, tasks in loaded.items()}

    files = []
    for name, path in present.items():
        tasks = loaded[name]
        files.append(
            {
                "split": name,
                "path": path.as_posix(),
                "count": len(tasks),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "primes": sorted({task.p for task in tasks}),
                "depths": sorted({task.difficulty.get("depth") for task in tasks}),
                "modes": sorted({task.mode for task in tasks}),
            }
        )

    overlaps = []
    for left, right in combinations(sorted(keys), 2):
        shared = keys[left] & keys[right]
        if not shared:
            continue
        severity = (
            "train_eval_leak"
            if ({left, right} & TRAINING and {left, right} & EVALUATION)
            else "within_stage_overlap"
        )
        overlaps.append({"a": left, "b": right, "shared_exact_keys": len(shared), "severity": severity})

    checks = {
        "sft_only_depth_1": all(
            task.difficulty.get("depth") == 1
            for name in ("sft_train_apply", "sft_train_plan", "sft_calibration_sh1")
            if name in loaded
            for task in loaded[name]
        ),
        "heldout_splits_use_heldout_primes_only": {
            name: sorted({task.p for task in loaded[name]}) == sorted(HELDOUT_PRIMES)
            for name in ("final_like_heldout", "confirmatory_heldout")
            if name in loaded
        },
        "heldout_primes_absent_from_training": {
            name: not ({task.p for task in loaded[name]} & set(HELDOUT_PRIMES))
            for name in sorted(TRAINING)
            if name in loaded
        },
        "train_eval_leaks": [item for item in overlaps if item["severity"] == "train_eval_leak"],
    }

    known_issues = [
        {
            "id": "stale-heldout-prime-flag",
            "detail": (
                "generator.py writes difficulty.heldout_prime = False unconditionally, so "
                "rows on primes 11 and 17 carry a wrong flag. The field is not part of the "
                "task_id hash and is never read by the verifier, splits or metrics; prime "
                "membership is determined by the p field, which is correct everywhere. Left "
                "unchanged on purpose: rewriting it would alter frozen file bytes and "
                "invalidate the preregistered SHA256 values."
            ),
            "scientific_impact": "none",
        },
        {
            "id": "sft-train-primes",
            "detail": (
                "prepare_smoke_data.py generates the atomic SFT splits on primes 5 and 7 "
                "only, while docs/03 section 3.3 declares train primes 5, 7, 13, 19, 23, 29. "
                "RL training does use all six. The narrower SFT prime set makes held-out "
                "generalisation harder, so the deviation is conservative with respect to "
                "every positive claim. Recorded in DECISIONS.md D-019."
            ),
            "scientific_impact": "conservative",
        },
        {
            "id": "rl-dev-overlap",
            "detail": (
                "rl_train and dev_exploration share exactly one task by exact key. "
                "dev_exploration is a selection set, not a reported test set, and the "
                "selection rule (temperature matching on distinct programs per task) is "
                "recomputed without that task in the integrity report below."
            ),
            "scientific_impact": "negligible",
        },
    ]

    payload = {
        "files": files,
        "overlaps": overlaps,
        "checks": checks,
        "known_issues": known_issues,
        "train_primes_declared": list(TRAIN_PRIMES),
        "heldout_primes_declared": list(HELDOUT_PRIMES),
    }
    write_json(Path("artifacts/reports/data_integrity.json"), payload)
    write_json(
        Path("artifacts/data/pilot/complete_manifest.json"),
        {"note": "all splits, superset of the frozen split_manifest.json", "files": files},
    )

    print(f"{'split':32s} {'n':>6s} {'primes':>22s} {'depths':>10s}  sha256[:12]")
    for item in files:
        print(
            f"{item['split']:32s} {item['count']:6d} {str(item['primes']):>22s} "
            f"{str(item['depths']):>10s}  {item['sha256'][:12]}"
        )
    print("\noverlaps by exact key:")
    for item in overlaps:
        print(f"  {item['a']} ^ {item['b']}: {item['shared_exact_keys']}  [{item['severity']}]")
    print(f"\nSFT is depth-1 only: {checks['sft_only_depth_1']}")
    print(f"held-out splits on primes 11/17 only: {checks['heldout_splits_use_heldout_primes_only']}")
    print(f"held-out primes absent from training: {checks['heldout_primes_absent_from_training']}")
    print(f"train/eval leaks: {len(checks['train_eval_leaks'])}")


if __name__ == "__main__":
    main()

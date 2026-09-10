"""D-021 data: can the model learn compositions from direct supervision?

Builds one training set of known-correct composition programs on train primes,
and a 2x2 evaluation grid that separates the two things every previous held-out
set changed at once:

    capacity_d2_train    depth 2, train primes      depth easy, primes seen
    capacity_d3_train    depth 3, train primes      depth hard, primes seen
    capacity_d2_heldout  depth 2, primes 11/17      depth easy, primes unseen
    capacity_d3_heldout  depth 3, primes 11/17      depth hard, primes unseen

Every generated task is rejected if its exact key already appears in any other
split, including the training set built here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbexp.experiment import sha256_file, write_json
from vbexp.generator import GenerationSpec, generate_many
from vbexp.io import read_tasks, write_tasks

DATA = Path("artifacts/data/pilot")
TRAIN_PRIMES = (5, 7, 13, 19, 23, 29)
HELDOUT_PRIMES = (11, 17)

EXISTING = [
    "sft_train_apply", "sft_train_plan", "sft_validation_apply",
    "sft_validation_apply_non_sh1", "sft_validation_plan", "sft_calibration_sh1",
    "rl_train", "dev_exploration", "final_like_heldout", "confirmatory_heldout",
    "confirmatory_heldout_v2",
]


def exact_key(task):
    return (
        task.mode, task.p, task.degree_cap, tuple(task.start),
        tuple(task.target) if task.target else None,
        tuple(task.program) if task.program else None, task.max_steps,
    )


def build(name: str, primes, depths, count: int, seed: int, taken: set) -> dict:
    spec = GenerationSpec(
        split=name, mode="plan", primes=tuple(primes), degree_caps=(2, 3),
        depths=tuple(depths), require_order_sensitive=True,
    )
    kept, rejected, batch, current = [], 0, count, seed
    while len(kept) < count:
        for task in generate_many(spec, batch, current):
            key = exact_key(task)
            if key in taken:
                rejected += 1
                continue
            taken.add(key)
            kept.append(task)
            if len(kept) == count:
                break
        current += 1
        batch = max(count - len(kept), 1)
    path = DATA / f"{name}.jsonl"
    write_tasks(path, kept)
    return {
        "split": name, "path": path.as_posix(), "count": len(kept),
        "size": path.stat().st_size, "sha256": sha256_file(path),
        "generator_seed": seed, "collisions_rejected": rejected,
        "primes": sorted({t.p for t in kept}),
        "depths": sorted({t.max_steps for t in kept}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--eval-count", type=int, default=300)
    args = parser.parse_args()

    taken = set()
    for name in EXISTING:
        path = DATA / f"{name}.jsonl"
        if path.exists():
            taken.update(exact_key(task) for task in read_tasks(path))

    records = [
        # Supervision: known-correct depth-2 and depth-3 programs, train primes only.
        build("sft_train_composition", TRAIN_PRIMES, (2, 3), args.train_count, 30001, taken),
        # 2x2 evaluation grid.
        build("capacity_d2_train", TRAIN_PRIMES, (2,), args.eval_count, 30011, taken),
        build("capacity_d3_train", TRAIN_PRIMES, (3,), args.eval_count, 30012, taken),
        build("capacity_d2_heldout", HELDOUT_PRIMES, (2,), args.eval_count, 30013, taken),
        build("capacity_d3_heldout", HELDOUT_PRIMES, (3,), args.eval_count, 30014, taken),
    ]
    write_json(DATA / "capacity_manifest.json", {"files": records})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

"""Generate a second, never-opened held-out PLAN depth-3 set (D-018).

`artifacts/data/pilot/final_like_heldout.jsonl` was already unsealed by the
July pilot, so re-using it cannot support a fresh confirmatory claim. This
script builds an independent set from the same generator specification with a
different generator seed, drops any task whose exact key collides with an
existing split, and freezes the SHA256.

Specification is identical to the original: PLAN, held-out primes 11 and 17,
degree caps 2 and 3, depth exactly 3, order-sensitivity required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbexp.experiment import sha256_file, write_json
from vbexp.generator import GenerationSpec, generate_many
from vbexp.io import read_tasks, write_tasks

EXISTING = [
    "artifacts/data/pilot/sft_train_apply.jsonl",
    "artifacts/data/pilot/sft_train_plan.jsonl",
    "artifacts/data/pilot/sft_validation_apply.jsonl",
    "artifacts/data/pilot/sft_validation_plan.jsonl",
    "artifacts/data/pilot/sft_calibration_sh1.jsonl",
    "artifacts/data/pilot/rl_train.jsonl",
    "artifacts/data/pilot/dev_exploration.jsonl",
    "artifacts/data/pilot/final_like_heldout.jsonl",
]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/pilot/confirmatory_heldout.jsonl"))
    args = parser.parse_args()

    taken = set()
    for path in EXISTING:
        candidate = Path(path)
        if candidate.exists():
            taken.update(exact_key(task) for task in read_tasks(candidate))

    spec = GenerationSpec(
        split="confirmatory_heldout",
        mode="plan",
        primes=(11, 17),
        degree_caps=(2, 3),
        depths=(3,),
        require_order_sensitive=True,
    )

    kept = []
    collisions = 0
    batch = args.count
    seed = args.seed
    while len(kept) < args.count:
        for task in generate_many(spec, batch, seed):
            key = exact_key(task)
            if key in taken:
                collisions += 1
                continue
            taken.add(key)
            kept.append(task)
            if len(kept) == args.count:
                break
        seed += 1
        batch = max(args.count - len(kept), 1)

    write_tasks(args.output, kept)
    digest = sha256_file(args.output)
    summary = {
        "path": args.output.as_posix(),
        "count": len(kept),
        "sha256": digest,
        "size": args.output.stat().st_size,
        "generator_seed": args.seed,
        "collisions_rejected": collisions,
        "primes": sorted({task.p for task in kept}),
        "depths": sorted({task.max_steps for task in kept}),
        "degree_caps": sorted({task.degree_cap for task in kept}),
    }
    write_json(Path("artifacts/data/pilot/confirmatory_heldout_manifest.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

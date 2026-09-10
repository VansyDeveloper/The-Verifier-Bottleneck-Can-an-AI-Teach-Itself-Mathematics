"""Degraded-verifier distillation sets: the project's titular axis, made testable.

`docs/01` defines verifier quality by alpha = P(accept | correct) and
beta = P(accept | wrong). `configs/experiments/verifier_cross.yaml` registers four
regimes, and none of them were ever run in this package. `artifacts/BLOCKERS.md`
argues Phase H is uninterpretable, and for the *self-training* endpoint that is
correct: with the GRPO effect indistinguishable from zero under a perfect
verifier, there is no effect left to degrade.

Direct composition supervision changed that. It produces a large, reliable
effect (+48 to +59 pp exhaustive Hit@32), so there now *is* something for a bad
verifier to destroy, and the question becomes well posed: how much of that gain
survives when the verifier that admitted the training trajectories has a
non-zero false-accept rate?

This script builds that intervention as data, not as new training code. For a
fraction beta of the 4000 distillation tasks, the supervised target is replaced
by a program of the same length that does *not* reach the target - exactly what a
verifier with false-accept rate beta would have waved through. alpha stays 1.0:
no correct trajectory is dropped, so the training budget and the example count
are identical across beta and only the label content changes.

Note that `modeling.target_text` reads `metadata["witness_program"]`, not
`task.program`, so that is the field the corruption must touch.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from vbexp.experiment import sha256_file, write_json
from vbexp.io import read_tasks, write_tasks
from vbexp.polynomial import OPERATIONS, apply_program
from vbexp.task import Task

DATA = Path("artifacts/data/pilot")
SOURCE = "sft_train_composition"


def wrong_program(rng: random.Random, task: Task, depth: int, attempts: int = 2000) -> tuple[str, ...]:
    """A same-length program that provably fails to reach the target."""
    target = tuple(task.target)
    allowed = tuple(task.operations) or tuple(OPERATIONS)
    for _ in range(attempts):
        candidate = tuple(rng.choice(allowed) for _ in range(depth))
        if apply_program(task.start, candidate, task.p) != target:
            return candidate
    raise RuntimeError(f"no incorrect depth-{depth} program found for {task.task_id}")


def corrupt(tasks: list[Task], beta: float, seed: int) -> tuple[list[Task], int]:
    rng = random.Random(seed)
    out: list[Task] = []
    corrupted = 0
    for task in tasks:
        witness = task.metadata.get("witness_program")
        if not witness:
            raise ValueError(f"task {task.task_id} has no witness_program")
        if rng.random() >= beta:
            out.append(task)
            continue
        replacement = wrong_program(rng, task, len(witness))
        metadata = {
            **task.metadata,
            "witness_program": list(replacement),
            "verifier_false_accept": True,
            "true_witness_program": list(witness),
        }
        out.append(_replace(task, metadata))
        corrupted += 1
    return out, corrupted


def _replace(task: Task, metadata: dict) -> Task:
    payload = task.to_dict()
    payload["metadata"] = metadata
    return Task.from_dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--seed", type=int, default=50021)
    args = parser.parse_args()

    tasks = read_tasks(DATA / f"{SOURCE}.jsonl")
    records = []
    for beta in args.betas:
        tag = f"{round(beta * 100):02d}"
        corrupted_tasks, count = corrupt(tasks, beta, args.seed + int(round(beta * 100)))
        name = f"sft_train_composition_noisy_b{tag}"
        path = DATA / f"{name}.jsonl"
        write_tasks(path, corrupted_tasks)
        # Independent audit of the written file rather than of the in-memory list.
        written = read_tasks(path)
        verified_wrong = sum(
            1
            for task in written
            if apply_program(task.start, tuple(task.metadata["witness_program"]), task.p)
            != tuple(task.target)
        )
        records.append(
            {
                "split": name,
                "path": path.as_posix(),
                "source": SOURCE,
                "alpha": 1.0,
                "beta_requested": beta,
                "count": len(written),
                "labels_corrupted": count,
                "beta_realised": count / len(written),
                "labels_verified_incorrect": verified_wrong,
                "sha256": sha256_file(path),
                "seed": args.seed + int(round(beta * 100)),
            }
        )
        if verified_wrong != count:
            raise RuntimeError(
                f"{name}: {count} labels marked corrupt but {verified_wrong} verify as incorrect"
            )

    write_json(DATA / "noisy_verifier_manifest.json", {"files": records})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

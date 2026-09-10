"""D-022 data: held-motif splits, crossed with held-out fields.

The external Stage-4 results show that novelty of *operation bigrams* ("motifs"),
not novelty of the field, is what composition supervision fails to transfer to.
Our D-021 grid varies depth and prime only, so it cannot see that axis at all.
This builds the missing one.

Held motifs, taken from the external protocol so the two studies are comparable:

    AX1 -> SH1,  AC1 -> REV,  SC2 -> AX1

A program *contains* a held motif if any adjacent pair matches. The four
evaluation families cross motif novelty with field novelty:

    motif_a  known fields, no held motif    (matches D-021 in-domain)
    motif_b  known fields, held motif required
    motif_c  held-out fields, no held motif
    motif_d  held-out fields, held motif required

Training data `sft_train_composition_nomotif` excludes every program containing a
held motif, so families B and D are genuinely unseen combinations built from
operations the model has individually mastered.

A task counts for a held-motif family only if *every* correct program of minimal
depth contains a held motif — otherwise the model could solve it by an unheld
route and the split would not test the motif at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbexp.experiment import sha256_file, write_json
from vbexp.generator import GenerationSpec, generate_many
from vbexp.io import read_tasks, write_tasks
from vbexp.search import all_solutions

DATA = Path("artifacts/data/pilot")
TRAIN_PRIMES = (5, 7, 13, 19, 23, 29)
HELDOUT_PRIMES = (11, 17)
HELD_MOTIFS = (("AX1", "SH1"), ("AC1", "REV"), ("SC2", "AX1"))

EXISTING = [
    "sft_train_apply", "sft_train_plan", "sft_validation_apply",
    "sft_validation_apply_non_sh1", "sft_validation_plan", "sft_calibration_sh1",
    "rl_train", "dev_exploration", "final_like_heldout", "confirmatory_heldout",
    "confirmatory_heldout_v2", "sft_train_composition",
    "capacity_d2_train", "capacity_d3_train", "capacity_d2_heldout", "capacity_d3_heldout",
]


def has_held_motif(program) -> bool:
    return any(pair in HELD_MOTIFS for pair in zip(program, program[1:]))


def exact_key(task):
    return (task.mode, task.p, task.degree_cap, tuple(task.start),
            tuple(task.target) if task.target else None,
            tuple(task.program) if task.program else None, task.max_steps)


def solutions_at_depth(task):
    return [p for p in all_solutions(task.start, task.target, task.p,
                                     task.operations, task.max_steps)
            if len(p) == task.max_steps]


def build(name, primes, depth, count, seed, taken, *, require_motif):
    """require_motif=True keeps only tasks whose every minimal solution uses a held motif."""
    spec = GenerationSpec(split=name, mode="plan", primes=tuple(primes), degree_caps=(2, 3),
                          depths=(depth,), require_order_sensitive=True)
    kept, rejected_key, rejected_motif, current = [], 0, 0, seed
    while len(kept) < count:
        for task in generate_many(spec, max(count - len(kept), 1) * 4, current):
            key = exact_key(task)
            if key in taken:
                rejected_key += 1
                continue
            sols = solutions_at_depth(task)
            if not sols:
                continue
            if require_motif and not all(has_held_motif(p) for p in sols):
                rejected_motif += 1
                continue
            if not require_motif and any(has_held_motif(p) for p in sols):
                rejected_motif += 1
                continue
            taken.add(key)
            kept.append(task)
            if len(kept) == count:
                break
        current += 1
    path = DATA / f"{name}.jsonl"
    write_tasks(path, kept)
    return {"split": name, "path": path.as_posix(), "count": len(kept),
            "sha256": sha256_file(path), "generator_seed": seed,
            "requires_held_motif": require_motif,
            "rejected_duplicate": rejected_key, "rejected_motif_filter": rejected_motif,
            "primes": sorted({t.p for t in kept}), "depth": depth}


def build_training(count, seed, taken):
    """Composition supervision with every held-motif program removed."""
    spec = GenerationSpec(split="sft_train_composition_nomotif", mode="plan",
                          primes=TRAIN_PRIMES, degree_caps=(2, 3), depths=(2, 3),
                          require_order_sensitive=True)
    kept, rejected, current = [], 0, seed
    while len(kept) < count:
        for task in generate_many(spec, max(count - len(kept), 1) * 3, current):
            key = exact_key(task)
            witness = tuple(task.metadata["witness_program"])
            if key in taken or has_held_motif(witness):
                rejected += 1
                continue
            taken.add(key)
            kept.append(task)
            if len(kept) == count:
                break
        current += 1
    path = DATA / "sft_train_composition_nomotif.jsonl"
    write_tasks(path, kept)
    return {"split": "sft_train_composition_nomotif", "path": path.as_posix(),
            "count": len(kept), "sha256": sha256_file(path), "generator_seed": seed,
            "rejected": rejected, "primes": sorted({t.p for t in kept}),
            "depths": sorted({t.max_steps for t in kept})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--eval-count", type=int, default=250)
    args = parser.parse_args()

    taken = set()
    for name in EXISTING:
        path = DATA / f"{name}.jsonl"
        if path.exists():
            taken.update(exact_key(task) for task in read_tasks(path))

    records = [
        build_training(args.train_count, 40000, taken),
        build("motif_a_d3", TRAIN_PRIMES, 3, args.eval_count, 40011, taken, require_motif=False),
        build("motif_b_d3", TRAIN_PRIMES, 3, args.eval_count, 40012, taken, require_motif=True),
        build("motif_c_d3", HELDOUT_PRIMES, 3, args.eval_count, 40013, taken, require_motif=False),
        build("motif_d_d3", HELDOUT_PRIMES, 3, args.eval_count, 40014, taken, require_motif=True),
    ]
    write_json(DATA / "motif_manifest.json",
               {"held_motifs": [list(m) for m in HELD_MOTIFS], "files": records})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

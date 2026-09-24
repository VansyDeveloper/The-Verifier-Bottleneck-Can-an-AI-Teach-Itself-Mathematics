"""Pair-conditioned exact held-out task generation."""

from __future__ import annotations

import random


def classify_programs(programs, excluded_pairs):
    flags = [any(pair in excluded_pairs for pair in zip(program, program[1:]))
             for program in programs]
    if not flags:
        raise ValueError("task has no correct programs")
    if all(flags):
        return "withheld"
    if not any(flags):
        return "control"
    return "mixed"


def generate_conditioned(core, registry, excluded_pairs, *, kind, count, seed, max_attempts=None):
    if kind not in ("withheld", "control") or count <= 0 or not excluded_pairs:
        raise ValueError("invalid holdout request")
    rng = random.Random(seed)
    semantic = core.semantics_for("A")
    witness_pool = [program for program in core._program_pool(3, "A")
                    if classify_programs((program,), excluded_pairs) == kind]
    if not witness_pool:
        raise ValueError(f"no {kind} witness programs")
    rows = []
    attempts = 0
    limit = max_attempts if max_attempts is not None else max(10000, count * 5000)
    split = f"{kind}_a"
    while len(rows) < count:
        attempts += 1
        if attempts > limit:
            raise RuntimeError(f"conditioned generation exhausted: {kind} {len(rows)}/{count}")
        p = rng.choice(semantic.fields)
        degree = rng.choice(core.DEGREES)
        start = tuple(rng.randrange(p) for _ in range(degree + 1))
        if not any(start):
            continue
        witness = rng.choice(witness_pool)
        states = core.trajectory(start, witness, p)
        if len(set(states)) != len(states):
            continue
        target = states[-1]
        search = core.exact_shortest_solutions(start, target, p, 3, limit=125)
        if search.shortest_depth != 3 or not core._search_has_clean_family(search, "A"):
            continue
        classification = classify_programs((solution.program for solution in search.solutions), excluded_pairs)
        if classification != kind and not (kind == "withheld" and classification == "mixed"):
            continue
        fingerprints = {core.canonical_state_fingerprint(p, state) for solution in search.solutions
                        for state in solution.states}
        if fingerprints & registry.states:
            continue
        task_fp = core.canonical_task_fingerprint(p, start, target, 3)
        if task_fp in registry.tasks:
            continue
        row = {
            "schema": core.TASK_SCHEMA, "task_id": core._task_id(task_fp),
            "task_fingerprint": task_fp, "split": split, "family": "A", "p": p,
            "depth": 3, "start": list(start), "target": list(target),
            "witness": list(witness), "states": [list(state) for state in states],
            "motif_count": core.motif_count(witness), "shortest_depth": 3,
            "shortest_solution_count": search.total_solutions,
            "solutions": [solution.as_record() for solution in search.solutions[:2]],
        }
        errors, _, validated_fp = core._validate_task_row(row, split, verify_shortest=False)
        if errors or validated_fp != task_fp:
            raise RuntimeError(f"generated task fails validation: {errors}")
        registry.states.update(fingerprints)
        registry.tasks.add(task_fp)
        rows.append(row)
    rng.shuffle(rows)
    return rows

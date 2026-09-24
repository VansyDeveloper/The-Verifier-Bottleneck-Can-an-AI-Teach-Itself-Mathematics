from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Sequence

from .polynomial import OPERATIONS, apply_program, is_order_sensitive
from .search import shortest_solutions
from .task import Task


@dataclass(frozen=True)
class GenerationSpec:
    split: str
    mode: str
    primes: tuple[int, ...]
    degree_caps: tuple[int, ...]
    depths: tuple[int, ...]
    operations: tuple[str, ...] = tuple(OPERATIONS)
    require_order_sensitive: bool = True
    require_unique_shortest: bool = False
    min_shortest_solutions: int = 1
    heldout_primes: tuple[int, ...] = ()


def _task_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def random_state(rng: random.Random, p: int, degree_cap: int) -> tuple[int, ...]:
    while True:
        state = tuple(rng.randrange(p) for _ in range(degree_cap + 1))
        if any(state):
            return state


def random_program(rng: random.Random, operations: Sequence[str], depth: int) -> tuple[str, ...]:
    if depth < 1:
        raise ValueError("depth must be positive")
    for _ in range(1000):
        program = tuple(rng.choice(operations) for _ in range(depth))
        if any(a == b == "REV" for a, b in zip(program, program[1:])):
            continue
        return program
    raise RuntimeError("unable to sample a valid program")


def generate_one(spec: GenerationSpec, rng: random.Random, max_attempts: int = 10000) -> Task:
    for _ in range(max_attempts):
        p = rng.choice(spec.primes)
        degree_cap = rng.choice(spec.degree_caps)
        depth = rng.choice(spec.depths)
        start = random_state(rng, p, degree_cap)
        program = random_program(rng, spec.operations, depth)
        target = apply_program(start, program, p)

        if spec.require_order_sensitive and depth > 1 and not is_order_sensitive(start, program, p):
            continue

        shortest = shortest_solutions(start, target, p, spec.operations, max_depth=depth)
        if not shortest or len(shortest[0]) != depth:
            continue
        if spec.require_unique_shortest and len(shortest) != 1:
            continue
        if len(shortest) < spec.min_shortest_solutions:
            continue

        payload = {
            "mode": spec.mode,
            "split": spec.split,
            "p": p,
            "degree_cap": degree_cap,
            "start": start,
            "target": target,
            "program": program,
            "max_steps": depth,
        }
        return Task(
            task_id=_task_id(payload),
            mode=spec.mode,  # type: ignore[arg-type]
            split=spec.split,
            p=p,
            degree_cap=degree_cap,
            start=start,
            operations=tuple(spec.operations),
            target=target,
            program=program if spec.mode == "apply" else None,
            max_steps=depth if spec.mode == "plan" else None,
            difficulty={"depth": depth, "heldout_prime": p in spec.heldout_primes},
            metadata={"witness_program": list(program), "shortest_solution_count": len(shortest)},
        )
    raise RuntimeError("unable to generate a task satisfying the requested constraints")


def generate_many(spec: GenerationSpec, count: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen: set[str] = set()
    while len(tasks) < count:
        task = generate_one(spec, rng)
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        tasks.append(task)
    return tasks

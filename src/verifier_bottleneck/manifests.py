from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MOD = 101


@dataclass(frozen=True)
class AffineMap:
    name: str
    a: int
    b: int

    def __post_init__(self) -> None:
        if self.a % MOD == 0:
            raise ValueError("Coefficient a must be non-zero modulo 101.")

        object.__setattr__(self, "a", self.a % MOD)
        object.__setattr__(self, "b", self.b % MOD)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "a": self.a,
            "b": self.b,
        }


def compose_affine_maps(maps_inner_to_outer: list[AffineMap]) -> tuple[int, int]:
    """
    Computes the canonical affine map (A, B) for a composition.

    If maps are [g1, g2, g3], this computes:

        g3(g2(g1(x)))

    Each map has the form:

        g(x) = a*x + b mod 101

    The result is:

        A*x + B mod 101
    """
    A = 1
    B = 0

    for g in maps_inner_to_outer:
        A = (g.a * A) % MOD
        B = (g.a * B + g.b) % MOD

    return A, B


def make_composition_expression(maps_inner_to_outer: list[AffineMap]) -> str:
    """
    For [g1, g2, g3], returns:

        g3(g2(g1(x)))
    """
    expr = "x"

    for g in maps_inner_to_outer:
        expr = f"{g.name}({expr})"

    return expr


def make_prompt(maps_inner_to_outer: list[AffineMap]) -> str:
    """
    Builds a model prompt for one task.
    """
    map_lines = []

    for g in maps_inner_to_outer:
        map_lines.append(f"{g.name}(x) = {g.a}*x + {g.b} mod {MOD}")

    maps_text = "\n".join(map_lines)
    composition_expr = make_composition_expression(maps_inner_to_outer)

    return (
        f"Over F_{MOD}, let\n"
        f"{maps_text}\n\n"
        f"Return the canonical affine map (A, B) for {composition_expr}.\n"
        f"That is, find A and B such that {composition_expr} = A*x + B mod {MOD}.\n"
        f"Final answer format: (A, B)."
    )


def make_task(task_id: str, length: int, rng: random.Random) -> dict:
    """
    Generates one task of a given composition length.

    length=1:
        g1(x)

    length=2:
        g2(g1(x))

    length=3:
        g3(g2(g1(x)))
    """
    if length < 1:
        raise ValueError("Task length must be at least 1.")

    maps = []

    for i in range(1, length + 1):
        a = rng.randint(1, MOD - 1)
        b = rng.randint(0, MOD - 1)
        maps.append(AffineMap(name=f"g{i}", a=a, b=b))

    A, B = compose_affine_maps(maps)
    prompt = make_prompt(maps)

    return {
        "id": task_id,
        "length": length,
        "prompt": prompt,
        "maps": [g.as_dict() for g in maps],
        "composition_order": [g.name for g in reversed(maps)],
        "answer": {
            "A": A,
            "B": B,
        },
        "answer_text": f"({A}, {B})",
    }


def prompt_key(task: dict) -> str:
    return task["prompt"]


def answer_key(task: dict) -> tuple[int, int]:
    return task["answer"]["A"], task["answer"]["B"]


def ordered_map_tuple_key(task: dict) -> tuple[tuple[int, int], ...]:
    """
    Represents the actual ordered maps in the composition.

    For g2(g1(x)), maps are stored inner-to-outer as [g1, g2],
    so the ordered tuple is:

        ((a1, b1), (a2, b2))
    """
    return tuple((m["a"], m["b"]) for m in task["maps"])


def has_collision(task: dict, used_prompts: set, used_answers: set, used_map_tuples: set) -> bool:
    return (
        prompt_key(task) in used_prompts
        or answer_key(task) in used_answers
        or ordered_map_tuple_key(task) in used_map_tuples
    )


def register_task(task: dict, used_prompts: set, used_answers: set, used_map_tuples: set) -> None:
    used_prompts.add(prompt_key(task))
    used_answers.add(answer_key(task))
    used_map_tuples.add(ordered_map_tuple_key(task))


def generate_manifest(
    *,
    prefix: str,
    length: int,
    n_tasks: int,
    rng: random.Random,
    used_prompts: set | None = None,
    used_answers: set | None = None,
    used_map_tuples: set | None = None,
) -> list[dict]:
    """
    Generates a manifest with no collisions against the provided used_* sets.

    This is important for source/heldout disjointness.
    """
    if used_prompts is None:
        used_prompts = set()

    if used_answers is None:
        used_answers = set()

    if used_map_tuples is None:
        used_map_tuples = set()

    tasks = []
    attempts = 0
    max_attempts = n_tasks * 1000

    while len(tasks) < n_tasks:
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate enough unique tasks for prefix={prefix}. "
                f"Generated {len(tasks)} out of {n_tasks}."
            )

        task_number = len(tasks)
        task_id = f"{prefix}_{task_number:06d}"
        task = make_task(task_id=task_id, length=length, rng=rng)

        attempts += 1

        if has_collision(task, used_prompts, used_answers, used_map_tuples):
            continue

        register_task(task, used_prompts, used_answers, used_map_tuples)
        tasks.append(task)

    return tasks


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_all_manifests(
    *,
    output_dir: str | Path = "data/manifests",
    seed: int = 42,
    n_warmup_l1: int = 100,
    n_source_l2: int = 500,
    n_heldout_l2: int = 200,
    n_transfer_l3: int = 100,
) -> dict[str, list[dict]]:
    """
    Builds all manifests.

    Important:
    - source_l2 and heldout_l2 share the same used_* sets,
      so they are disjoint by prompt, canonical answer, and ordered map tuple.
    - warmup_l1 and transfer_l3 are separate auxiliary splits.
    """
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    warmup_l1 = generate_manifest(
        prefix="warmup_l1",
        length=1,
        n_tasks=n_warmup_l1,
        rng=rng,
    )

    used_prompts_l2: set[str] = set()
    used_answers_l2: set[tuple[int, int]] = set()
    used_map_tuples_l2: set[tuple[tuple[int, int], ...]] = set()

    source_l2 = generate_manifest(
        prefix="source_l2",
        length=2,
        n_tasks=n_source_l2,
        rng=rng,
        used_prompts=used_prompts_l2,
        used_answers=used_answers_l2,
        used_map_tuples=used_map_tuples_l2,
    )

    heldout_l2 = generate_manifest(
        prefix="heldout_l2",
        length=2,
        n_tasks=n_heldout_l2,
        rng=rng,
        used_prompts=used_prompts_l2,
        used_answers=used_answers_l2,
        used_map_tuples=used_map_tuples_l2,
    )

    transfer_l3 = generate_manifest(
        prefix="transfer_l3",
        length=3,
        n_tasks=n_transfer_l3,
        rng=rng,
    )

    manifests = {
        "warmup_l1": warmup_l1,
        "source_l2": source_l2,
        "heldout_l2": heldout_l2,
        "transfer_l3": transfer_l3,
    }

    for name, rows in manifests.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    return manifests


def validate_no_overlap(source: list[dict], heldout: list[dict]) -> None:
    """
    Checks that source_l2 and heldout_l2 are disjoint.

    We check:
    - prompt overlap
    - canonical answer overlap
    - ordered map tuple overlap
    """
    source_prompts = {prompt_key(t) for t in source}
    heldout_prompts = {prompt_key(t) for t in heldout}

    source_answers = {answer_key(t) for t in source}
    heldout_answers = {answer_key(t) for t in heldout}

    source_map_tuples = {ordered_map_tuple_key(t) for t in source}
    heldout_map_tuples = {ordered_map_tuple_key(t) for t in heldout}

    prompt_overlap = source_prompts & heldout_prompts
    answer_overlap = source_answers & heldout_answers
    map_tuple_overlap = source_map_tuples & heldout_map_tuples

    if prompt_overlap:
        raise ValueError(f"source/heldout prompt overlap: {len(prompt_overlap)}")

    if answer_overlap:
        raise ValueError(f"source/heldout answer overlap: {len(answer_overlap)}")

    if map_tuple_overlap:
        raise ValueError(f"source/heldout ordered map tuple overlap: {len(map_tuple_overlap)}")


def validate_manifest(tasks: list[dict]) -> None:
    """
    Basic sanity checks for one manifest.
    """
    ids = [t["id"] for t in tasks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task ids found.")

    for task in tasks:
        maps = [
            AffineMap(name=m["name"], a=m["a"], b=m["b"])
            for m in task["maps"]
        ]

        A, B = compose_affine_maps(maps)
        gold = task["answer"]

        if A != gold["A"] or B != gold["B"]:
            raise ValueError(
                f"Wrong answer in task {task['id']}: "
                f"expected ({A}, {B}), got ({gold['A']}, {gold['B']})"
            )


def validate_all_manifests(manifest_dir: str | Path = "data/manifests") -> None:
    manifest_dir = Path(manifest_dir)

    warmup_l1 = read_jsonl(manifest_dir / "warmup_l1.jsonl")
    source_l2 = read_jsonl(manifest_dir / "source_l2.jsonl")
    heldout_l2 = read_jsonl(manifest_dir / "heldout_l2.jsonl")
    transfer_l3 = read_jsonl(manifest_dir / "transfer_l3.jsonl")

    for tasks in [warmup_l1, source_l2, heldout_l2, transfer_l3]:
        validate_manifest(tasks)

    validate_no_overlap(source_l2, heldout_l2)
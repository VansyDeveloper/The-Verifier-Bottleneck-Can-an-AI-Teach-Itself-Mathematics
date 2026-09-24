from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

from .polynomial import State, apply_program


def iter_programs(operations: Sequence[str], max_depth: int, min_depth: int = 0) -> Iterable[tuple[str, ...]]:
    if max_depth < 0 or min_depth < 0 or min_depth > max_depth:
        raise ValueError("invalid depth range")
    for depth in range(min_depth, max_depth + 1):
        for program in product(operations, repeat=depth):
            yield tuple(program)


def all_solutions(
    start: Sequence[int],
    target: Sequence[int],
    p: int,
    operations: Sequence[str],
    max_depth: int,
) -> list[tuple[str, ...]]:
    target_tuple: State = tuple(int(v) % p for v in target)
    return [
        program
        for program in iter_programs(operations, max_depth=max_depth, min_depth=0)
        if apply_program(start, program, p) == target_tuple
    ]


def shortest_solutions(
    start: Sequence[int],
    target: Sequence[int],
    p: int,
    operations: Sequence[str],
    max_depth: int,
) -> list[tuple[str, ...]]:
    target_tuple: State = tuple(int(v) % p for v in target)
    for depth in range(max_depth + 1):
        solutions = [
            program
            for program in iter_programs(operations, max_depth=depth, min_depth=depth)
            if apply_program(start, program, p) == target_tuple
        ]
        if solutions:
            return solutions
    return []

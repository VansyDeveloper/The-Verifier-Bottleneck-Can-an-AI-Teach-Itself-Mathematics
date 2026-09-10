from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Mapping, Sequence

import composition_core as core


ALGORITHM = "deterministic_exhaustive_cell_fallback_v1"


def _state_from_lexicographic_index(index: int, p: int, width: int) -> tuple[int, ...]:
    """Map ``0..p**width-1`` to product(range(p), repeat=width) order."""

    values = [0] * width
    for position in range(width - 1, -1, -1):
        values[position] = index % p
        index //= p
    return tuple(values)


def _row_state_fingerprints(row: Mapping[str, object]) -> set[str]:
    p = int(row["p"])
    states: list[Sequence[int]] = list(row.get("states", ()))  # type: ignore[arg-type]
    for solution in row.get("solutions", ()):  # type: ignore[assignment]
        if isinstance(solution, Mapping):
            states.extend(solution.get("states", ()))  # type: ignore[arg-type]
    if not states:
        states = list(core.trajectory(row["start"], row.get("witness") or (row["operation"],), p))  # type: ignore[arg-type]
    return {core.canonical_state_fingerprint(p, state) for state in states}


def _reserve_rows(
    registry: core.FingerprintRegistry,
    splits: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    for rows in splits.values():
        for row in rows:
            registry.states.update(_row_state_fingerprints(row))
            registry.tasks.add(core.task_fingerprint(row))


def generate_confirm_atomic_data_v11(
    atomic_reference: core.AtomicReference,
    pilot_splits: Mapping[str, Sequence[Mapping[str, object]]],
    final_splits: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    per_operation: int = 200,
    seed: int = 73200,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate the non-gating atomic set with deterministic capacity fallback.

    The legacy requested field/degree schedule is retained. Candidate states in
    each cell are scanned exactly once in lexicographic order. When the requested
    cell has no remaining exact, disjoint candidate, the other cells are tried in
    one seed-derived cyclic order fixed before inspecting candidate validity. No
    model score, generation, metric, or alternative seed is consulted.
    """

    if per_operation <= 0:
        raise ValueError("per_operation must be positive")
    seed = int(seed)
    cells = [(int(p), int(degree)) for p in core.KNOWN_FIELDS for degree in core.DEGREES]
    registry = core.FingerprintRegistry.from_atomic_reference(atomic_reference)
    _reserve_rows(registry, pilot_splits)
    _reserve_rows(registry, final_splits)

    rows: list[dict[str, object]] = []
    requested_counts: Counter[tuple[int, int]] = Counter()
    realized_counts: Counter[tuple[int, int]] = Counter()
    realized_by_operation: dict[str, Counter[tuple[int, int]]] = {
        operation: Counter() for operation in core.OPS
    }
    scan_counts: Counter[tuple[str, int, int]] = Counter()
    exhausted: set[tuple[str, int, int]] = set()
    fallback_events: list[dict[str, object]] = []

    for operation_index, operation in enumerate(core.OPS):
        schedule = [cells[index % len(cells)] for index in range(per_operation)]
        random.Random(seed + 10_000 + operation_index).shuffle(schedule)
        cursors = {cell: 0 for cell in cells}

        for row_index, requested_cell in enumerate(schedule):
            requested_counts[requested_cell] += 1
            remaining = [cell for cell in cells if cell != requested_cell]
            random.Random(seed + 20_000 + operation_index * per_operation + row_index).shuffle(remaining)
            fallback_order = [requested_cell, *remaining]
            accepted: tuple[dict[str, object], tuple[int, int]] | None = None

            for p, degree in fallback_order:
                cell = (p, degree)
                total = p ** (degree + 1)
                while cursors[cell] < total:
                    candidate_index = cursors[cell]
                    cursors[cell] += 1
                    scan_counts[(operation, p, degree)] += 1
                    start = _state_from_lexicographic_index(candidate_index, p, degree + 1)
                    if not any(start):
                        continue
                    states = core.trajectory(start, (operation,), p)
                    if states[-1] == states[0]:
                        continue
                    state_fingerprints = {
                        core.canonical_state_fingerprint(p, state) for state in states
                    }
                    if state_fingerprints & registry.states:
                        continue
                    task_fingerprint = core.canonical_task_fingerprint(p, states[0], states[-1], 1)
                    if task_fingerprint in registry.tasks:
                        continue
                    search = core.exact_shortest_solutions(states[0], states[-1], p, 1, limit=2)
                    if search.shortest_depth != 1 or not search.solutions:
                        continue
                    row: dict[str, object] = {
                        "schema": core.TASK_SCHEMA,
                        "task_id": "s4c-" + task_fingerprint[:24],
                        "task_fingerprint": task_fingerprint,
                        "split": "confirm_atomic",
                        "family": "ATOMIC",
                        "p": p,
                        "degree": degree,
                        "depth": 1,
                        "operation": operation,
                        "start": list(states[0]),
                        "target": list(states[-1]),
                        "witness": [operation],
                        "states": [list(state) for state in states],
                        "motif_count": 0,
                        "shortest_depth": 1,
                        "shortest_solution_count": search.total_solutions,
                        "solutions": [solution.as_record() for solution in search.solutions],
                    }
                    registry.states.update(state_fingerprints)
                    for solution in search.solutions:
                        registry.states.update(
                            core.canonical_state_fingerprint(p, state) for state in solution.states
                        )
                    registry.tasks.add(task_fingerprint)
                    accepted = (row, cell)
                    break
                if accepted is not None:
                    break
                exhausted.add((operation, p, degree))

            if accepted is None:
                raise RuntimeError(
                    f"confirm atomic v1.1 exhausted every field/degree cell for {operation} row {row_index}"
                )
            row, realized_cell = accepted
            rows.append(row)
            realized_counts[realized_cell] += 1
            realized_by_operation[operation][realized_cell] += 1
            if realized_cell != requested_cell:
                fallback_events.append(
                    {
                        "operation": operation,
                        "row_index": row_index,
                        "requested": {"p": requested_cell[0], "degree": requested_cell[1]},
                        "realized": {"p": realized_cell[0], "degree": realized_cell[1]},
                    }
                )

    random.Random(seed).shuffle(rows)
    operation_counts = Counter(str(row["operation"]) for row in rows)
    expected = Counter({operation: per_operation for operation in core.OPS})
    if operation_counts != expected:
        raise core.CompositionIntegrityError("confirm atomic v1.1 operation balance failed")
    audit = core.audit_splits({"confirm_atomic": rows}, atomic_reference=atomic_reference)
    if not audit["ok"]:
        raise core.CompositionIntegrityError(f"confirm atomic v1.1 audit failed: {audit['errors']}")

    def cell_key(cell: tuple[int, int]) -> str:
        return f"p{cell[0]}_degree{cell[1]}"

    generation_audit: dict[str, object] = {
        "schema": "stage4.composition.confirmation.confirm-atomic-generation.v1",
        "status": "PASS",
        "algorithm": ALGORITHM,
        "seed": seed,
        "per_operation": per_operation,
        "rows": len(rows),
        "operation_counts": dict(sorted(operation_counts.items())),
        "requested_cell_counts": {
            cell_key(cell): requested_counts[cell] for cell in sorted(requested_counts)
        },
        "realized_cell_counts": {
            cell_key(cell): realized_counts[cell] for cell in sorted(realized_counts)
        },
        "realized_by_operation": {
            operation: {
                cell_key(cell): realized_by_operation[operation][cell]
                for cell in sorted(realized_by_operation[operation])
            }
            for operation in core.OPS
        },
        "fallback_count": len(fallback_events),
        "fallback_events": fallback_events,
        "candidate_scan_counts": {
            f"{operation}:{cell_key((p, degree))}": count
            for (operation, p, degree), count in sorted(scan_counts.items())
        },
        "exhausted_cells": [
            {"operation": operation, "p": p, "degree": degree}
            for operation, p, degree in sorted(exhausted)
        ],
        "output": core.digest_jsonl_rows(rows),
        "atomic_reference_audit": audit,
        "model_outputs_consulted": False,
        "alternative_seeds_tried": 0,
    }
    return rows, generation_audit


__all__ = ["ALGORITHM", "generate_confirm_atomic_data_v11"]

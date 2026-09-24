from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Mapping, Sequence


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_SIZE_LOCK = "0.6B"

OPS = ("SH1", "SC2", "REV", "AC1", "AX1")
TOKENS = {op: f"<OP{i}>" for i, op in enumerate(OPS)}
INV_TOKENS = {token: op for op, token in TOKENS.items()}
HELDOUT_MOTIFS = (("AX1", "SH1"), ("AC1", "REV"), ("SC2", "AX1"))
HELD_MOTIFS = HELDOUT_MOTIFS

KNOWN_FIELDS = (5, 7, 13, 19, 23, 29)
TRANSFER_FIELDS = (11, 17)
DEGREES = (2, 3, 4)
TRAIN_DEPTH_COUNTS = {2: 1000, 3: 2000, 4: 1000}

TASK_SCHEMA = "stage4.composition.v5.task.v1"
TRAJECTORY_SCHEMA = "stage4.composition.v5.trajectory.v1"


class CompositionIntegrityError(ValueError):
    """Raised when hash-bound inputs or generated records fail integrity checks."""


@dataclass(frozen=True)
class SplitSemantics:
    family: str
    fields: tuple[int, ...]
    motif_rule: str


SPLIT_SEMANTICS = {
    "ATOMIC": SplitSemantics("ATOMIC", KNOWN_FIELDS, "none"),
    "TRAIN": SplitSemantics("TRAIN", KNOWN_FIELDS, "none"),
    "A": SplitSemantics("A", KNOWN_FIELDS, "none"),
    "B": SplitSemantics("B", KNOWN_FIELDS, "exactly_one"),
    "C": SplitSemantics("C", TRANSFER_FIELDS, "none"),
    "D": SplitSemantics("D", TRANSFER_FIELDS, "exactly_one"),
}


def _require_prime(p: int) -> int:
    p = int(p)
    if p < 2 or any(p % divisor == 0 for divisor in range(2, math.isqrt(p) + 1)):
        raise ValueError(f"field modulus must be prime, got {p}")
    return p


def _normal_state(state: Sequence[int], p: int) -> tuple[int, ...]:
    p = _require_prime(p)
    out = tuple(int(value) % p for value in state)
    if not out:
        raise ValueError("state must contain at least one coefficient")
    return out


def _normal_program(program: Sequence[str]) -> tuple[str, ...]:
    out = tuple(str(op) for op in program)
    if not out or any(op not in OPS for op in out):
        raise ValueError(f"program must contain only registered operations: {out!r}")
    return out


def _fingerprint(domain: str, value: object) -> str:
    payload = json.dumps(
        {"domain": domain, "value": value},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_state_fingerprint(p: int, state: Sequence[int]) -> str:
    """Return a split-independent fingerprint for one normalized field state."""

    p = _require_prime(p)
    return _fingerprint("stage4.composition.state.v1", [p, list(_normal_state(state, p))])


def state_fingerprint(p: int, state: Sequence[int]) -> str:
    """Integration alias for :func:`canonical_state_fingerprint`."""

    return canonical_state_fingerprint(p, state)


def canonical_task_fingerprint(
    p: int,
    start: Sequence[int],
    target: Sequence[int],
    depth: int,
) -> str:
    """Fingerprint the semantic task, intentionally excluding split, witness, and nonce."""

    p = _require_prime(p)
    depth = int(depth)
    if depth not in (1, 2, 3, 4):
        raise ValueError(f"depth must be 1..4, got {depth}")
    return _fingerprint(
        "stage4.composition.task.v1",
        [p, list(_normal_state(start, p)), list(_normal_state(target, p)), depth],
    )


def task_fingerprint(row: Mapping[str, object]) -> str:
    program = row.get("witness") or row.get("program")
    depth = int(row.get("depth") or (len(program) if program else 0))
    return canonical_task_fingerprint(
        int(row["p"]),
        row["start"],  # type: ignore[arg-type]
        row["target"],  # type: ignore[arg-type]
        depth,
    )


def canonical_trajectory_fingerprint(
    p: int,
    start: Sequence[int],
    target: Sequence[int],
    program: Sequence[str],
) -> str:
    p = _require_prime(p)
    return _fingerprint(
        "stage4.composition.trajectory.v1",
        [
            p,
            list(_normal_state(start, p)),
            list(_normal_state(target, p)),
            list(_normal_program(program)),
        ],
    )


def apply_op(state: Sequence[int], op: str, p: int) -> tuple[int, ...]:
    """Apply one registered exact polynomial-coefficient operation over GF(p)."""

    p = _require_prime(p)
    state = _normal_state(state, p)
    if op == "SH1":
        result = [0] * len(state)
        for i, coefficient in enumerate(state):
            for j in range(i + 1):
                result[j] = (result[j] + coefficient * math.comb(i, j)) % p
        return tuple(result)
    if op == "SC2":
        return tuple(coefficient * pow(2, i, p) % p for i, coefficient in enumerate(state))
    if op == "REV":
        return tuple(reversed(state))
    if op in ("AC1", "AX1"):
        index = 0 if op == "AC1" else 1
        if index >= len(state):
            raise ValueError(f"{op} requires a state with at least {index + 1} coefficients")
        result = list(state)
        result[index] = (result[index] + 1) % p
        return tuple(result)
    raise ValueError(f"unknown operation: {op}")


def trajectory(start: Sequence[int], program: Sequence[str], p: int) -> tuple[tuple[int, ...], ...]:
    p = _require_prime(p)
    program = _normal_program(program)
    states = [_normal_state(start, p)]
    for op in program:
        states.append(apply_op(states[-1], op, p))
    return tuple(states)


def verify_program(start: Sequence[int], target: Sequence[int], program: Sequence[str], p: int) -> bool:
    p = _require_prime(p)
    return trajectory(start, program, p)[-1] == _normal_state(target, p)


def format_state(state: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in state) + "]"


def plan_prompt(row: Mapping[str, object]) -> str:
    depth = int(row.get("depth") or len(_row_program(row)))
    return (
        f"FIELD: {int(row['p'])}\n"
        f"DEGREE_CAP: {len(row['start']) - 1}\n"  # type: ignore[arg-type]
        f"START: {format_state(row['start'])}\n"  # type: ignore[arg-type]
        f"TARGET: {format_state(row['target'])}\n"  # type: ignore[arg-type]
        f"ALLOWED: {' '.join(TOKENS[op] for op in OPS)}\n"
        f"MAX_STEPS: {depth}\n"
        "Return exactly one line:\nPROGRAM:"
    )


def apply_prompt(row: Mapping[str, object]) -> str:
    program = _row_program(row)
    return (
        f"FIELD: {int(row['p'])}\n"
        f"DEGREE_CAP: {len(row['start']) - 1}\n"  # type: ignore[arg-type]
        f"START: {format_state(row['start'])}\n"  # type: ignore[arg-type]
        f"PROGRAM: {' '.join(TOKENS[op] for op in program)}\n"
        "Return exactly one line:\nRESULT:"
    )


def program_answer(program: Sequence[str]) -> str:
    return " ".join(TOKENS[op] for op in _normal_program(program))


@lru_cache(maxsize=4)
def enumerate_programs(depth: int) -> tuple[tuple[str, ...], ...]:
    depth = int(depth)
    if depth not in (1, 2, 3, 4):
        raise ValueError(f"depth must be 1..4, got {depth}")
    return tuple(itertools.product(OPS, repeat=depth))


def motif_count(program: Sequence[str]) -> int:
    program = _normal_program(program)
    return sum(tuple(program[index : index + 2]) in HELDOUT_MOTIFS for index in range(len(program) - 1))


def semantics_for(family: str) -> SplitSemantics:
    family = str(family).upper()
    try:
        return SPLIT_SEMANTICS[family]
    except KeyError as exc:
        raise ValueError(f"unknown split family {family!r}; expected one of {tuple(SPLIT_SEMANTICS)}") from exc


def program_matches_family(program: Sequence[str], family: str) -> bool:
    rule = semantics_for(family).motif_rule
    count = motif_count(program)
    return count == 0 if rule == "none" else count == 1


@dataclass(frozen=True)
class ExactSolution:
    program: tuple[str, ...]
    states: tuple[tuple[int, ...], ...]

    def as_record(self) -> dict[str, object]:
        return {"program": list(self.program), "states": [list(state) for state in self.states]}


@dataclass(frozen=True)
class ShortestSolutionSearch:
    shortest_depth: int | None
    solutions: tuple[ExactSolution, ...]
    total_solutions: int
    motif_histogram: Mapping[int, int]


def exact_shortest_solutions(
    start: Sequence[int],
    target: Sequence[int],
    p: int,
    max_depth: int,
    *,
    limit: int = 2,
) -> ShortestSolutionSearch:
    """Enumerate exact programs and retain at most ``limit`` solutions at the shortest depth."""

    if limit < 1:
        raise ValueError("limit must be positive")
    if int(max_depth) not in (1, 2, 3, 4):
        raise ValueError("max_depth must be 1..4")
    p = _require_prime(p)
    start = _normal_state(start, p)
    target = _normal_state(target, p)
    for depth in range(1, int(max_depth) + 1):
        kept: list[ExactSolution] = []
        histogram: Counter[int] = Counter()
        total = 0
        for program in enumerate_programs(depth):
            states = trajectory(start, program, p)
            if states[-1] != target:
                continue
            total += 1
            histogram[motif_count(program)] += 1
            if len(kept) < limit:
                kept.append(ExactSolution(program, states))
        if total:
            return ShortestSolutionSearch(depth, tuple(kept), total, dict(sorted(histogram.items())))
    return ShortestSolutionSearch(None, (), 0, {})


def _search_has_clean_family(search: ShortestSolutionSearch, family: str) -> bool:
    if search.shortest_depth is None or not search.motif_histogram:
        return False
    wanted = 0 if semantics_for(family).motif_rule == "none" else 1
    return set(search.motif_histogram) == {wanted}


def canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def digest_jsonl_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    payload = canonical_jsonl_bytes(rows)
    return {"rows": len(rows), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()}


def _manifest_entries(manifest: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("files", "data", "entries"):
        nested = manifest.get(key)
        if isinstance(nested, Mapping):
            return nested
    return manifest


def _manifest_entry(entries: Mapping[str, object], split: str) -> Mapping[str, object] | None:
    for key in (split, f"{split}.jsonl"):
        value = entries.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _row_program(row: Mapping[str, object]) -> tuple[str, ...]:
    if row.get("program"):
        return _normal_program(row["program"])  # type: ignore[arg-type]
    if row.get("witness"):
        return _normal_program(row["witness"])  # type: ignore[arg-type]
    if row.get("operation"):
        return _normal_program((str(row["operation"]),))
    raise CompositionIntegrityError("row has no program, witness, or operation")


def _exact_row_states(row: Mapping[str, object]) -> tuple[tuple[int, ...], ...]:
    p = int(row["p"])
    program = _row_program(row)
    states = trajectory(row["start"], program, p)  # type: ignore[arg-type]
    if states[-1] != _normal_state(row["target"], p):  # type: ignore[arg-type]
        raise CompositionIntegrityError("row target does not match exact verifier")
    stored = row.get("states")
    if stored is not None:
        normalized = tuple(_normal_state(state, p) for state in stored)  # type: ignore[arg-type]
        if normalized != states:
            raise CompositionIntegrityError("stored intermediate states do not match exact trajectory")
    return states


def _fingerprint_digest(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


@dataclass(frozen=True)
class AtomicReference:
    state_fingerprints_by_split: Mapping[str, frozenset[str]]
    task_fingerprints_by_split: Mapping[str, frozenset[str]]
    forbidden_state_fingerprints: frozenset[str]
    forbidden_task_fingerprints: frozenset[str]
    report: Mapping[str, object]

    @property
    def ok(self) -> bool:
        return bool(self.report.get("ok"))


def audit_atomic_reference(
    digest_manifest: Mapping[str, object],
    raw_splits: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    required_splits: Sequence[str] = ("calibration", "final"),
    strict: bool = True,
) -> AtomicReference:
    """Bind parsed atomic rows to their JSONL digest manifest and expose raw-state exclusions."""

    entries = _manifest_entries(digest_manifest)
    errors: list[str] = []
    digest_checks: dict[str, object] = {}
    state_sets: dict[str, frozenset[str]] = {}
    task_sets: dict[str, frozenset[str]] = {}
    for required in required_splits:
        if required not in raw_splits:
            errors.append(f"required atomic split is absent: {required}")
    for split, rows in sorted(raw_splits.items()):
        entry = _manifest_entry(entries, split)
        actual = digest_jsonl_rows(rows)
        checks = {
            "manifest_present": entry is not None,
            "rows_match": bool(entry) and int(entry.get("rows", -1)) == actual["rows"],
            "bytes_match": bool(entry) and int(entry.get("bytes", -1)) == actual["bytes"],
            "sha256_match": bool(entry)
            and str(entry.get("sha256", "")).upper() == str(actual["sha256"]).upper(),
        }
        digest_checks[split] = {"actual": actual, "checks": checks}
        if not all(checks.values()):
            errors.append(f"atomic digest mismatch for {split}: {checks}")
        state_fps: set[str] = set()
        task_fps: set[str] = set()
        for index, row in enumerate(rows):
            try:
                states = _exact_row_states(row)
                p = int(row["p"])
                state_fps.update(canonical_state_fingerprint(p, state) for state in states)
                task_fps.add(
                    canonical_task_fingerprint(p, states[0], states[-1], len(_row_program(row)))
                )
            except Exception as exc:
                errors.append(f"invalid atomic row {split}[{index}]: {exc}")
        state_sets[split] = frozenset(state_fps)
        task_sets[split] = frozenset(task_fps)
    state_overlap: dict[str, int] = {}
    task_overlap: dict[str, int] = {}
    for left, right in itertools.combinations(sorted(raw_splits), 2):
        state_overlap[f"{left}:{right}"] = len(state_sets.get(left, frozenset()) & state_sets.get(right, frozenset()))
        task_overlap[f"{left}:{right}"] = len(task_sets.get(left, frozenset()) & task_sets.get(right, frozenset()))
    all_states = frozenset().union(*state_sets.values()) if state_sets else frozenset()
    all_tasks = frozenset().union(*task_sets.values()) if task_sets else frozenset()
    report = {
        "schema": "stage4.composition.v5.atomic-reference-audit.v1",
        "model": MODEL_ID,
        "digest_checks": digest_checks,
        "counts": {
            split: {"rows": len(raw_splits[split]), "raw_states": len(state_sets[split]), "tasks": len(task_sets[split])}
            for split in sorted(raw_splits)
        },
        "state_overlap_between_atomic_splits": state_overlap,
        "task_overlap_between_atomic_splits": task_overlap,
        "all_raw_state_fingerprint_sha256": _fingerprint_digest(all_states),
        "all_task_fingerprint_sha256": _fingerprint_digest(all_tasks),
        "errors": errors,
        "ok": not errors,
    }
    reference = AtomicReference(state_sets, task_sets, all_states, all_tasks, report)
    if strict and not reference.ok:
        raise CompositionIntegrityError("; ".join(errors))
    return reference


@dataclass
class FingerprintRegistry:
    states: set[str] = field(default_factory=set)
    tasks: set[str] = field(default_factory=set)

    @classmethod
    def from_atomic_reference(cls, reference: AtomicReference) -> "FingerprintRegistry":
        if not reference.ok:
            raise CompositionIntegrityError("cannot seed generation from a failed atomic reference audit")
        return cls(set(reference.forbidden_state_fingerprints), set(reference.forbidden_task_fingerprints))


@lru_cache(maxsize=20)
def _program_pool(depth: int, family: str) -> tuple[tuple[str, ...], ...]:
    return tuple(program for program in enumerate_programs(depth) if program_matches_family(program, family))


def _row_state_fingerprints(row: Mapping[str, object]) -> set[str]:
    p = int(row["p"])
    states = {_normal_state(state, p) for state in row.get("states", ())}  # type: ignore[arg-type]
    for solution in row.get("solutions", ()):  # type: ignore[assignment]
        if isinstance(solution, Mapping):
            states.update(_normal_state(state, p) for state in solution.get("states", ()))  # type: ignore[arg-type]
    if not states:
        states.update(_exact_row_states(row))
    return {canonical_state_fingerprint(p, state) for state in states}


def _task_id(task_fp: str) -> str:
    return "s4c-" + task_fp[:24]


def generate_split(
    name: str,
    count: int,
    family: str,
    *,
    seed: int,
    depths: Sequence[int] = (3,),
    depth_counts: Mapping[int, int] | None = None,
    registry: FingerprintRegistry | None = None,
    degrees: Sequence[int] = DEGREES,
    require_shortest_depth: bool = True,
    max_attempts: int | None = None,
) -> list[dict[str, object]]:
    """Generate one deterministic in-memory split without writing scientific data."""

    if count <= 0:
        raise ValueError("count must be positive")
    semantic = semantics_for(family)
    depths = tuple(int(depth) for depth in depths)
    if any(depth not in (2, 3, 4) for depth in depths):
        raise ValueError("composition depths must be 2, 3, or 4")
    degrees = tuple(int(degree) for degree in degrees)
    if any(degree not in DEGREES for degree in degrees):
        raise ValueError(f"degrees must be chosen from {DEGREES}")
    rng = random.Random(int(seed))
    if depth_counts is not None:
        normalized_counts = {int(depth): int(value) for depth, value in depth_counts.items()}
        if sum(normalized_counts.values()) != count or any(value < 0 for value in normalized_counts.values()):
            raise ValueError("depth_counts must be non-negative and sum exactly to count")
        if any(depth not in depths for depth in normalized_counts):
            raise ValueError("depth_counts contains a depth absent from depths")
        schedule = [depth for depth in sorted(normalized_counts) for _ in range(normalized_counts[depth])]
        rng.shuffle(schedule)
    else:
        schedule = [rng.choice(depths) for _ in range(count)]
    registry = registry if registry is not None else FingerprintRegistry()
    rows: list[dict[str, object]] = []
    attempts = 0
    limit = max_attempts if max_attempts is not None else max(10_000, count * 5_000)
    while len(rows) < count:
        attempts += 1
        if attempts > limit:
            raise RuntimeError(f"generation exhausted for {name}: {len(rows)}/{count} after {attempts - 1} attempts")
        depth = schedule[len(rows)]
        p = rng.choice(semantic.fields)
        degree = rng.choice(degrees)
        start = tuple(rng.randrange(p) for _ in range(degree + 1))
        if not any(start):
            continue
        pool = _program_pool(depth, semantic.family)
        if not pool:
            raise RuntimeError(f"no programs satisfy family {semantic.family} at depth {depth}")
        witness = rng.choice(pool)
        witness_states = trajectory(start, witness, p)
        if len(set(witness_states)) != len(witness_states):
            continue
        target = witness_states[-1]
        initial_state_fps = {canonical_state_fingerprint(p, state) for state in witness_states}
        if initial_state_fps & registry.states:
            continue
        search = exact_shortest_solutions(start, target, p, depth, limit=2)
        if search.shortest_depth is None:
            continue
        if require_shortest_depth and search.shortest_depth != depth:
            continue
        if not _search_has_clean_family(search, semantic.family):
            continue
        solution_records = [solution.as_record() for solution in search.solutions]
        all_state_fps = set(initial_state_fps)
        for solution in search.solutions:
            all_state_fps.update(canonical_state_fingerprint(p, state) for state in solution.states)
        if all_state_fps & registry.states:
            continue
        task_fp = canonical_task_fingerprint(p, start, target, depth)
        if task_fp in registry.tasks:
            continue
        row: dict[str, object] = {
            "schema": TASK_SCHEMA,
            "task_id": _task_id(task_fp),
            "task_fingerprint": task_fp,
            "split": str(name),
            "family": semantic.family,
            "p": p,
            "depth": depth,
            "start": list(start),
            "target": list(target),
            "witness": list(witness),
            "states": [list(state) for state in witness_states],
            "motif_count": motif_count(witness),
            "shortest_depth": search.shortest_depth,
            "shortest_solution_count": search.total_solutions,
            "solutions": solution_records,
        }
        registry.states.update(all_state_fps)
        registry.tasks.add(task_fp)
        rows.append(row)
    rng.shuffle(rows)
    return rows


def _expected_family(split: str, declared: str) -> str:
    normalized = split.lower().replace("-", "_")
    if normalized == "train":
        return "TRAIN"
    for family in ("a", "b", "c", "d"):
        if normalized.endswith("_" + family):
            return family.upper()
    return declared.upper()


def _validate_task_row(row: Mapping[str, object], split: str, *, verify_shortest: bool) -> tuple[list[str], set[str], str | None]:
    errors: list[str] = []
    try:
        if row.get("schema") != TASK_SCHEMA:
            errors.append(f"unexpected schema {row.get('schema')!r}")
        if row.get("split") != split:
            errors.append(f"row split {row.get('split')!r} does not match container {split!r}")
        family = str(row["family"]).upper()
        if family != _expected_family(split, family):
            errors.append(f"family {family} does not match split {split}")
        semantic = semantics_for(family)
        p = int(row["p"])
        if p not in semantic.fields:
            errors.append(f"field {p} is invalid for family {family}")
        depth = int(row["depth"])
        witness = _normal_program(row["witness"])  # type: ignore[arg-type]
        if len(witness) != depth:
            errors.append("witness length does not equal depth")
        if not program_matches_family(witness, family):
            errors.append("witness violates split motif semantics")
        states = trajectory(row["start"], witness, p)  # type: ignore[arg-type]
        if states[-1] != _normal_state(row["target"], p):  # type: ignore[arg-type]
            errors.append("witness fails exact verifier")
        stored_states = tuple(_normal_state(state, p) for state in row["states"])  # type: ignore[arg-type]
        if stored_states != states:
            errors.append("stored witness states are not exact")
        expected_fp = canonical_task_fingerprint(p, states[0], states[-1], depth)
        if row.get("task_fingerprint") != expected_fp:
            errors.append("task_fingerprint mismatch")
        if row.get("task_id") != _task_id(expected_fp):
            errors.append("task_id mismatch")
        solutions = row.get("solutions")
        if not isinstance(solutions, Sequence) or isinstance(solutions, (str, bytes)) or len(solutions) > 2:
            errors.append("solutions must contain at most two records")
        if verify_shortest:
            search = exact_shortest_solutions(states[0], states[-1], p, depth, limit=2)
            if row.get("shortest_depth") != search.shortest_depth:
                errors.append("shortest_depth mismatch")
            if row.get("shortest_solution_count") != search.total_solutions:
                errors.append("shortest_solution_count mismatch")
            expected_solutions = [solution.as_record() for solution in search.solutions]
            if solutions != expected_solutions:
                errors.append("stored shortest solutions mismatch exact search")
            if not _search_has_clean_family(search, family):
                errors.append("shortest solutions violate split motif semantics")
        return errors, _row_state_fingerprints(row), expected_fp
    except Exception as exc:
        errors.append(str(exc))
        return errors, set(), None


def audit_splits(
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    atomic_reference: AtomicReference | None = None,
    verify_shortest: bool = True,
) -> dict[str, object]:
    """Audit exactness, fingerprints, motif leakage, and state/task disjointness."""

    errors: list[str] = []
    states_by_split: dict[str, set[str]] = {}
    tasks_by_split: dict[str, set[str]] = {}
    within_state_overlap: dict[str, int] = {}
    heldout_train_leaks: list[str] = []
    for split, rows in sorted(splits.items()):
        states: set[str] = set()
        tasks: set[str] = set()
        state_owner: dict[str, str] = {}
        duplicate_states = 0
        for index, row in enumerate(rows):
            row_errors, row_states, row_task = _validate_task_row(row, split, verify_shortest=verify_shortest)
            errors.extend(f"{split}[{index}]: {error}" for error in row_errors)
            task_id = str(row.get("task_id", f"row-{index}"))
            for state_fp in row_states:
                owner = state_owner.get(state_fp)
                if owner is not None and owner != task_id:
                    duplicate_states += 1
                else:
                    state_owner[state_fp] = task_id
            states.update(row_states)
            if row_task is not None:
                if row_task in tasks:
                    errors.append(f"{split}[{index}]: duplicate canonical task")
                tasks.add(row_task)
            if split.lower() == "train":
                programs = [row.get("witness", ())]
                programs.extend(
                    solution.get("program", ())
                    for solution in row.get("solutions", ())  # type: ignore[assignment]
                    if isinstance(solution, Mapping)
                )
                if any(motif_count(program) != 0 for program in programs):  # type: ignore[arg-type]
                    heldout_train_leaks.append(task_id)
        states_by_split[split] = states
        tasks_by_split[split] = tasks
        within_state_overlap[split] = duplicate_states
        if duplicate_states:
            errors.append(f"{split}: {duplicate_states} state fingerprints are shared by different tasks")
    state_overlap: dict[str, int] = {}
    task_overlap: dict[str, int] = {}
    for left, right in itertools.combinations(sorted(splits), 2):
        state_overlap[f"{left}:{right}"] = len(states_by_split[left] & states_by_split[right])
        task_overlap[f"{left}:{right}"] = len(tasks_by_split[left] & tasks_by_split[right])
    atomic_state_overlap: dict[str, int] = {}
    atomic_task_overlap: dict[str, int] = {}
    if atomic_reference is not None:
        if not atomic_reference.ok:
            errors.append("atomic reference audit is not OK")
        for split in sorted(splits):
            atomic_state_overlap[split] = len(states_by_split[split] & atomic_reference.forbidden_state_fingerprints)
            atomic_task_overlap[split] = len(tasks_by_split[split] & atomic_reference.forbidden_task_fingerprints)
    if heldout_train_leaks:
        errors.append(f"train contains {len(heldout_train_leaks)} held-motif labels")
    if any(state_overlap.values()):
        errors.append("state fingerprints overlap between composition splits")
    if any(task_overlap.values()):
        errors.append("task fingerprints overlap between composition splits")
    if any(atomic_state_overlap.values()):
        errors.append("composition states overlap atomic raw states")
    if any(atomic_task_overlap.values()):
        errors.append("composition tasks overlap atomic tasks")
    return {
        "schema": "stage4.composition.v5.split-audit.v1",
        "model": MODEL_ID,
        "counts": {split: len(rows) for split, rows in sorted(splits.items())},
        "within_split_state_overlap": within_state_overlap,
        "state_overlap": state_overlap,
        "task_overlap": task_overlap,
        "atomic_state_overlap": atomic_state_overlap,
        "atomic_task_overlap": atomic_task_overlap,
        "train_heldout_motif_leaks": heldout_train_leaks,
        "errors": errors,
        "ok": not errors,
    }


def generate_pilot_splits(
    atomic_reference: AtomicReference,
    *,
    train_depth_counts: Mapping[int, int] = TRAIN_DEPTH_COUNTS,
    dev_a_count: int = 500,
    dev_b_count: int = 500,
    train_seed: int = 42000,
    dev_a_seed: int = 42001,
    dev_b_seed: int = 42002,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Generate preregistered pilot splits in memory; this function never creates final data."""

    normalized = {int(depth): int(count) for depth, count in train_depth_counts.items()}
    if set(normalized) != {2, 3, 4} or sum(normalized.values()) < 4000:
        raise ValueError("pilot train requires depths 2/3/4 and at least 4000 tasks")
    if dev_a_count < 500 or dev_b_count < 500:
        raise ValueError("pilot dev-A and dev-B require at least 500 tasks each")
    registry = FingerprintRegistry.from_atomic_reference(atomic_reference)
    train = generate_split(
        "train",
        sum(normalized.values()),
        "TRAIN",
        seed=train_seed,
        depths=(2, 3, 4),
        depth_counts=normalized,
        registry=registry,
    )
    dev_a = generate_split("dev_a", dev_a_count, "A", seed=dev_a_seed, depths=(3,), registry=registry)
    dev_b = generate_split("dev_b", dev_b_count, "B", seed=dev_b_seed, depths=(3,), registry=registry)
    splits = {"train": train, "dev_a": dev_a, "dev_b": dev_b}
    audit = audit_splits(splits, atomic_reference=atomic_reference)
    if not audit["ok"]:
        raise CompositionIntegrityError(f"generated pilot split audit failed: {audit['errors']}")
    return splits, audit


@dataclass(frozen=True)
class DiscoveryThresholds:
    min_unique_trajectories: int = 2000
    min_solvable_tasks: int = 1000
    min_full_signatures: int = 20
    min_position_count: int = 25
    max_first_operation_spread: int = 1


def _trajectory_record(row: Mapping[str, object], program: Sequence[str], states: Sequence[Sequence[int]]) -> dict[str, object]:
    p = int(row["p"])
    program = _normal_program(program)
    normalized_states = tuple(_normal_state(state, p) for state in states)
    target = _normal_state(row["target"], p)  # type: ignore[arg-type]
    if normalized_states[0] != _normal_state(row["start"], p) or normalized_states[-1] != target:  # type: ignore[arg-type]
        raise CompositionIntegrityError("candidate trajectory endpoints do not match task")
    if normalized_states != trajectory(normalized_states[0], program, p):
        raise CompositionIntegrityError("candidate trajectory fails exact verifier")
    fingerprint = canonical_trajectory_fingerprint(p, normalized_states[0], target, program)
    return {
        "schema": TRAJECTORY_SCHEMA,
        "trajectory_id": "s4t-" + fingerprint[:24],
        "trajectory_fingerprint": fingerprint,
        "task_id": row["task_id"],
        "split": row["split"],
        "p": p,
        "depth": len(program),
        "start": list(normalized_states[0]),
        "target": list(target),
        "program": list(program),
        "states": [list(state) for state in normalized_states],
        "motif_count": motif_count(program),
    }


def _discovery_candidates(train_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    candidates: dict[str, dict[str, object]] = {}
    for row in train_rows:
        if str(row.get("family", "")).upper() != "TRAIN":
            raise CompositionIntegrityError("Discover accepts TRAIN rows only")
        witness = _normal_program(row["witness"])  # type: ignore[arg-type]
        records = [(witness, row["states"])]  # type: ignore[list-item]
        records.extend(
            (_normal_program(solution["program"]), solution["states"])  # type: ignore[arg-type]
            for solution in row.get("solutions", ())  # type: ignore[assignment]
            if isinstance(solution, Mapping)
        )
        for program, states in records:
            if motif_count(program):
                raise CompositionIntegrityError(f"held motif leaked into train task {row.get('task_id')}")
            record = _trajectory_record(row, program, states)
            candidates[str(record["trajectory_fingerprint"])] = record
    return sorted(candidates.values(), key=lambda row: str(row["trajectory_fingerprint"]))


def balance_discovered_trajectories(
    train_rows: Sequence[Mapping[str, object]],
    *,
    target_count: int = 2250,
    thresholds: DiscoveryThresholds = DiscoveryThresholds(),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select a deterministic, one-task-per-trajectory balanced Discover corpus."""

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    candidates = _discovery_candidates(train_rows)
    pools: defaultdict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    depths = sorted({int(row["depth"]) for row in candidates})
    for row in candidates:
        pools[(int(row["depth"]), str(row["program"][0]))].append(row)  # type: ignore[index]
    for pool in pools.values():
        pool.sort(key=lambda row: str(row["trajectory_fingerprint"]))
    expected_pools = [(depth, op) for depth in depths for op in OPS]
    quota_base, quota_remainder = divmod(target_count, len(OPS))
    first_operation_target = {
        op: quota_base + (index < quota_remainder)
        for index, op in enumerate(OPS)
    }
    selected: list[dict[str, object]] = []
    used_tasks: set[str] = set()
    selected_first_operations: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    local_motifs: Counter[str] = Counter()
    signatures: Counter[str] = Counter()

    def score(row: Mapping[str, object]) -> tuple[float, float, int, str]:
        program = tuple(row["program"])  # type: ignore[arg-type]
        position_pressure = sum(position_counts[f"{index}:{op}"] for index, op in enumerate(program)) / len(program)
        motifs = [f"{program[index]}>{program[index + 1]}" for index in range(len(program) - 1)]
        motif_pressure = sum(local_motifs[motif] for motif in motifs) / max(1, len(motifs))
        signature = " ".join(program)
        return position_pressure, motif_pressure, signatures[signature], str(row["trajectory_fingerprint"])

    while len(selected) < target_count:
        progress = False
        for key in expected_pools:
            _depth, first_op = key
            if selected_first_operations[first_op] >= first_operation_target[first_op]:
                continue
            eligible = [row for row in pools.get(key, ()) if str(row["task_id"]) not in used_tasks]
            if not eligible:
                continue
            chosen = min(eligible, key=score)
            selected.append(chosen)
            used_tasks.add(str(chosen["task_id"]))
            program = tuple(chosen["program"])  # type: ignore[arg-type]
            selected_first_operations[str(program[0])] += 1
            signatures[" ".join(program)] += 1
            for index, op in enumerate(program):
                position_counts[f"{index}:{op}"] += 1
            for index in range(len(program) - 1):
                local_motifs[f"{program[index]}>{program[index + 1]}"] += 1
            progress = True
            if len(selected) >= target_count:
                break
        if not progress:
            break
    max_depth = max((int(row["depth"]) for row in selected), default=0)
    required_positions = [f"{index}:{op}" for index in range(max_depth) for op in OPS]
    minimum_position = min((position_counts.get(key, 0) for key in required_positions), default=0)
    first_counts = Counter(str(row["program"][0]) for row in selected)  # type: ignore[index]
    first_spread = (max(first_counts.values()) - min(first_counts.values())) if len(first_counts) == len(OPS) else target_count
    first_target_met = all(first_counts.get(op, 0) == first_operation_target[op] for op in OPS)
    motif_leaks = [str(row["trajectory_id"]) for row in selected if int(row["motif_count"]) != 0]
    unique_trajectories = len({str(row["trajectory_fingerprint"]) for row in selected})
    solvable_tasks = len(used_tasks)
    full_signatures = len(signatures)
    checks = {
        "unique_trajectories": unique_trajectories >= thresholds.min_unique_trajectories,
        "solvable_tasks": solvable_tasks >= thresholds.min_solvable_tasks,
        "full_signatures": full_signatures >= thresholds.min_full_signatures,
        "position_coverage": minimum_position >= thresholds.min_position_count,
        "first_operation_balance": first_target_met and first_spread <= thresholds.max_first_operation_spread,
        "held_motif_zero": not motif_leaks,
    }
    report = {
        "schema": "stage4.composition.v5.discover-audit.v1",
        "candidate_trajectories": len(candidates),
        "unique_trajectories": unique_trajectories,
        "solvable_tasks": solvable_tasks,
        "full_signatures": full_signatures,
        "position_operation": dict(sorted(position_counts.items())),
        "minimum_position_count": minimum_position,
        "first_operation": dict(sorted(first_counts.items())),
        "first_operation_target": dict(sorted(first_operation_target.items())),
        "first_operation_spread": first_spread,
        "local_motifs": dict(sorted(local_motifs.items())),
        "held_motif_leaks": motif_leaks,
        "thresholds": {
            "min_unique_trajectories": thresholds.min_unique_trajectories,
            "min_solvable_tasks": thresholds.min_solvable_tasks,
            "min_full_signatures": thresholds.min_full_signatures,
            "min_position_count": thresholds.min_position_count,
            "max_first_operation_spread": thresholds.max_first_operation_spread,
        },
        "checks": checks,
        "passes_minimums": all(checks.values()),
    }
    return selected, report


def generate_pilot_data(
    atomic_reference: AtomicReference,
    **kwargs: object,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Stable integration entry point for in-memory pilot generation."""

    return generate_pilot_splits(atomic_reference, **kwargs)  # type: ignore[arg-type]


def exact_discover(
    train_rows: Sequence[Mapping[str, object]],
    *,
    target_count: int = 2250,
    thresholds: DiscoveryThresholds = DiscoveryThresholds(),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Stable integration entry point for exact, balanced Discover selection."""

    return balance_discovered_trajectories(
        train_rows,
        target_count=target_count,
        thresholds=thresholds,
    )


def audit_pilot_data(
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    atomic_reference: AtomicReference,
    verify_shortest: bool = True,
) -> dict[str, object]:
    """Stable integration entry point for the pilot exclusion/integrity audit."""

    return audit_splits(
        splits,
        atomic_reference=atomic_reference,
        verify_shortest=verify_shortest,
    )


def _reserve_existing_rows(
    registry: FingerprintRegistry,
    splits: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    for rows in splits.values():
        for row in rows:
            registry.states.update(_row_state_fingerprints(row))
            registry.tasks.add(task_fingerprint(row))


def generate_final_data(
    atomic_reference: AtomicReference,
    pilot_splits: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    config_frozen: bool,
    depth3_sizes: Mapping[str, int] | None = None,
    depth2_probe_per_split: int = 125,
    depth4_probe_per_split: int = 125,
    seed_base: int = 50010,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Generate final rows only in memory and only after an external freeze guard passes.

    The orchestration layer remains responsible for verifying and creating
    ``CONFIG_FROZEN.json``. Merely importing this module never creates final data.
    """

    if not config_frozen:
        raise CompositionIntegrityError("final data generation is locked until config_frozen=True")
    pilot_audit = audit_splits(pilot_splits, atomic_reference=atomic_reference)
    if not pilot_audit["ok"]:
        raise CompositionIntegrityError(f"cannot generate final data from invalid pilot splits: {pilot_audit['errors']}")
    sizes = dict(depth3_sizes or {"A": 1000, "B": 1000, "C": 500, "D": 500})
    if set(sizes) != {"A", "B", "C", "D"} or any(int(size) <= 0 for size in sizes.values()):
        raise ValueError("depth3_sizes must contain positive A/B/C/D counts")
    if depth2_probe_per_split <= 0 or depth4_probe_per_split <= 0:
        raise ValueError("depth probe counts must be positive")
    registry = FingerprintRegistry.from_atomic_reference(atomic_reference)
    _reserve_existing_rows(registry, pilot_splits)
    generated: dict[str, list[dict[str, object]]] = {}
    seed = int(seed_base)
    for family in ("A", "B", "C", "D"):
        name = f"final_{family.lower()}"
        generated[name] = generate_split(
            name,
            int(sizes[family]),
            family,
            seed=seed,
            depths=(3,),
            registry=registry,
        )
        seed += 1
        for depth, count in ((2, depth2_probe_per_split), (4, depth4_probe_per_split)):
            probe_name = f"{name}_depth{depth}"
            generated[probe_name] = generate_split(
                probe_name,
                int(count),
                family,
                seed=seed,
                depths=(depth,),
                registry=registry,
            )
            seed += 1
    combined: dict[str, Sequence[Mapping[str, object]]] = dict(pilot_splits)
    combined.update(generated)
    audit = audit_splits(combined, atomic_reference=atomic_reference)
    if not audit["ok"]:
        raise CompositionIntegrityError(f"generated final split audit failed: {audit['errors']}")
    return generated, audit


def generate_confirm_atomic_data(
    atomic_reference: AtomicReference,
    pilot_splits: Mapping[str, Sequence[Mapping[str, object]]],
    final_splits: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    per_operation: int = 200,
    seed: int = 50100,
    max_attempts_per_cell: int = 20_000,
) -> list[dict[str, object]]:
    """Create a fresh balanced atomic forgetting set in memory after final freeze.

    ``depth`` remains the program depth (one); ``degree`` records polynomial
    state degree 2/3/4. The orchestration layer owns the freeze guard and file
    creation, while this pure function guarantees exactness and exclusions.
    """

    if per_operation <= 0:
        raise ValueError("per_operation must be positive")
    if max_attempts_per_cell <= 0:
        raise ValueError("max_attempts_per_cell must be positive")
    registry = FingerprintRegistry.from_atomic_reference(atomic_reference)
    _reserve_existing_rows(registry, pilot_splits)
    _reserve_existing_rows(registry, final_splits)
    rng = random.Random(int(seed))
    cells = [(p, degree) for p in KNOWN_FIELDS for degree in DEGREES]
    rows: list[dict[str, object]] = []
    for op_index, op in enumerate(OPS):
        schedule = [cells[index % len(cells)] for index in range(per_operation)]
        random.Random(int(seed) + 10_000 + op_index).shuffle(schedule)
        for p, degree in schedule:
            for _ in range(max_attempts_per_cell):
                start = tuple(rng.randrange(p) for _ in range(degree + 1))
                if not any(start):
                    continue
                states = trajectory(start, (op,), p)
                if states[-1] == states[0]:
                    continue
                state_fps = {canonical_state_fingerprint(p, state) for state in states}
                if state_fps & registry.states:
                    continue
                task_fp = canonical_task_fingerprint(p, states[0], states[-1], 1)
                if task_fp in registry.tasks:
                    continue
                search = exact_shortest_solutions(states[0], states[-1], p, 1, limit=2)
                if search.shortest_depth != 1 or not search.solutions:
                    continue
                row: dict[str, object] = {
                    "schema": TASK_SCHEMA,
                    "task_id": _task_id(task_fp),
                    "task_fingerprint": task_fp,
                    "split": "confirm_atomic",
                    "family": "ATOMIC",
                    "p": p,
                    "degree": degree,
                    "depth": 1,
                    "operation": op,
                    "start": list(states[0]),
                    "target": list(states[-1]),
                    "witness": [op],
                    "states": [list(state) for state in states],
                    "motif_count": 0,
                    "shortest_depth": 1,
                    "shortest_solution_count": search.total_solutions,
                    "solutions": [solution.as_record() for solution in search.solutions],
                }
                registry.states.update(state_fps)
                for solution in search.solutions:
                    registry.states.update(canonical_state_fingerprint(p, state) for state in solution.states)
                registry.tasks.add(task_fp)
                rows.append(row)
                break
            else:
                raise RuntimeError(f"confirm atomic generation exhausted for {op}, p={p}, degree={degree}")
    rng.shuffle(rows)
    if Counter(str(row["operation"]) for row in rows) != Counter({op: per_operation for op in OPS}):
        raise CompositionIntegrityError("confirm atomic operation balance failed")
    audit = audit_splits({"confirm_atomic": rows}, atomic_reference=atomic_reference)
    if not audit["ok"]:
        raise CompositionIntegrityError(f"confirm atomic audit failed: {audit['errors']}")
    return rows


__all__ = [
    "MODEL_ID",
    "MODEL_SIZE_LOCK",
    "OPS",
    "TOKENS",
    "INV_TOKENS",
    "HELDOUT_MOTIFS",
    "HELD_MOTIFS",
    "KNOWN_FIELDS",
    "TRANSFER_FIELDS",
    "TRAIN_DEPTH_COUNTS",
    "TASK_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "SPLIT_SEMANTICS",
    "CompositionIntegrityError",
    "SplitSemantics",
    "ExactSolution",
    "ShortestSolutionSearch",
    "AtomicReference",
    "FingerprintRegistry",
    "DiscoveryThresholds",
    "apply_op",
    "trajectory",
    "verify_program",
    "enumerate_programs",
    "motif_count",
    "semantics_for",
    "program_matches_family",
    "canonical_state_fingerprint",
    "state_fingerprint",
    "canonical_task_fingerprint",
    "task_fingerprint",
    "canonical_trajectory_fingerprint",
    "exact_shortest_solutions",
    "canonical_jsonl_bytes",
    "digest_jsonl_rows",
    "audit_atomic_reference",
    "generate_split",
    "audit_splits",
    "generate_pilot_splits",
    "balance_discovered_trajectories",
    "format_state",
    "plan_prompt",
    "apply_prompt",
    "program_answer",
    "generate_pilot_data",
    "exact_discover",
    "audit_pilot_data",
    "generate_final_data",
    "generate_confirm_atomic_data",
]

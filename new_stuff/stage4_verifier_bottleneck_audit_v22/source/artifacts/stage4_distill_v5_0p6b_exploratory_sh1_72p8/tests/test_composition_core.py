from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import composition_core as core


def _atomic_row(split: str, start: tuple[int, ...], op: str, *, p: int = 29) -> dict[str, object]:
    states = core.trajectory(start, (op,), p)
    return {
        "schema": "stage4.sh1.v5.task.v1",
        "task_id": f"atomic-{split}-{op}-{'-'.join(map(str, start))}",
        "split": split,
        "p": p,
        "start": list(start),
        "target": list(states[-1]),
        "operation": op,
        "program": [op],
        "witness": [op],
    }


def _atomic_reference() -> core.AtomicReference:
    raw = {
        "calibration": [_atomic_row("calibration", (1, 2, 3, 4), "AC1")],
        "final": [_atomic_row("final", (7, 8, 9, 10), "SC2")],
    }
    manifest = {f"{split}.jsonl": core.digest_jsonl_rows(rows) for split, rows in raw.items()}
    return core.audit_atomic_reference(manifest, raw)


def _synthetic_task(index: int, depth: int, op: str) -> dict[str, object]:
    p = 29
    start = ((index + 1) % p, (index * 3 + 2) % p, (index * 5 + 3) % p, (index * 7 + 4) % p, 1)
    program = (op,) * depth
    states = core.trajectory(start, program, p)
    fingerprint = core.canonical_task_fingerprint(p, start, states[-1], depth)
    return {
        "schema": core.TASK_SCHEMA,
        "task_id": "s4c-" + fingerprint[:24],
        "task_fingerprint": fingerprint,
        "split": "train",
        "family": "TRAIN",
        "p": p,
        "depth": depth,
        "start": list(start),
        "target": list(states[-1]),
        "witness": list(program),
        "states": [list(state) for state in states],
        "motif_count": 0,
        "shortest_depth": depth,
        "shortest_solution_count": 1,
        "solutions": [{"program": list(program), "states": [list(state) for state in states]}],
    }


def test_model_operation_and_token_locks() -> None:
    assert core.MODEL_ID == "Qwen/Qwen3-0.6B"
    assert core.MODEL_SIZE_LOCK == "0.6B"
    assert core.OPS == ("SH1", "SC2", "REV", "AC1", "AX1")
    assert core.TOKENS == {"SH1": "<OP0>", "SC2": "<OP1>", "REV": "<OP2>", "AC1": "<OP3>", "AX1": "<OP4>"}
    assert core.HELD_MOTIFS == (("AX1", "SH1"), ("AC1", "REV"), ("SC2", "AX1"))


def test_exact_operations_trajectory_and_verifier() -> None:
    state = (1, 2, 3)
    assert core.apply_op(state, "SH1", 5) == (1, 3, 3)
    assert core.apply_op(state, "SC2", 5) == (1, 4, 2)
    assert core.apply_op(state, "REV", 5) == (3, 2, 1)
    assert core.apply_op(state, "AC1", 5) == (2, 2, 3)
    assert core.apply_op(state, "AX1", 5) == (1, 3, 3)
    states = core.trajectory(state, ("AC1", "REV"), 5)
    assert states == ((1, 2, 3), (2, 2, 3), (3, 2, 2))
    assert core.verify_program(state, states[-1], ("AC1", "REV"), 5)
    assert not core.verify_program(state, state, ("AC1", "REV"), 5)
    with pytest.raises(ValueError, match="prime"):
        core.apply_op(state, "REV", 6)
    with pytest.raises(ValueError, match="unknown operation"):
        core.apply_op(state, "NOPE", 5)


def test_exact_enumeration_counts_order_and_uniqueness() -> None:
    programs = {depth: core.enumerate_programs(depth) for depth in (2, 3, 4)}
    assert {depth: len(rows) for depth, rows in programs.items()} == {2: 25, 3: 125, 4: 625}
    assert programs[2][0] == ("SH1", "SH1")
    assert programs[2][-1] == ("AX1", "AX1")
    assert all(len(set(rows)) == len(rows) for rows in programs.values())


def test_canonical_fingerprints_ignore_split_and_normalize_field_values() -> None:
    assert core.state_fingerprint(5, (1, 2, 3)) == core.canonical_state_fingerprint(5, (6, 7, 8))
    first = core.canonical_task_fingerprint(5, (1, 2, 3), (2, 2, 3), 2)
    row = {"p": 5, "start": [1, 2, 3], "target": [2, 2, 3], "depth": 2, "split": "dev_a", "witness": ["AC1", "REV"]}
    changed_split = {**row, "split": "final_c", "witness": ["REV", "AC1"]}
    assert core.task_fingerprint(row) == first
    assert core.task_fingerprint(changed_split) == first
    assert first != core.canonical_task_fingerprint(5, row["start"], row["target"], 3)


def test_exact_shortest_search_keeps_at_most_two_with_states() -> None:
    start = (1, 2, 3)
    target = core.trajectory(start, ("AC1", "AX1"), 5)[-1]
    search = core.exact_shortest_solutions(start, target, 5, 2, limit=2)
    assert search.shortest_depth == 2
    assert search.total_solutions == 4
    assert len(search.solutions) == 2
    assert search.motif_histogram == {0: 4}
    for solution in search.solutions:
        assert len(solution.states) == 3
        assert solution.states == core.trajectory(start, solution.program, 5)
        assert solution.states[-1] == target


def test_split_family_semantics_are_the_registered_two_by_two_design() -> None:
    assert core.semantics_for("A").fields == core.KNOWN_FIELDS
    assert core.semantics_for("B").fields == core.KNOWN_FIELDS
    assert core.semantics_for("C").fields == core.TRANSFER_FIELDS
    assert core.semantics_for("D").fields == core.TRANSFER_FIELDS
    held = ("AX1", "SH1", "REV")
    clean = ("SH1", "REV", "AC1")
    assert core.motif_count(held) == 1
    assert core.program_matches_family(held, "B")
    assert core.program_matches_family(held, "D")
    assert not core.program_matches_family(held, "A")
    assert core.program_matches_family(clean, "A")
    assert core.program_matches_family(clean, "C")


def test_prompt_helpers_preserve_registered_single_token_contract() -> None:
    row = {"p": 5, "depth": 2, "start": [1, 2, 3], "target": [3, 2, 2], "program": ["AC1", "REV"]}
    plan = core.plan_prompt(row)
    apply = core.apply_prompt(row)
    assert plan.endswith("Return exactly one line:\nPROGRAM:")
    assert "ALLOWED: <OP0> <OP1> <OP2> <OP3> <OP4>" in plan
    assert "PROGRAM: <OP3> <OP2>" in apply
    assert core.program_answer(("AC1", "REV")) == "<OP3> <OP2>"
    assert core.format_state((1, 2, 3)) == "[1, 2, 3]"


def test_atomic_reference_binds_manifest_and_raw_exact_states() -> None:
    calibration = [_atomic_row("calibration", (1, 2, 3, 4), "AC1")]
    # Deliberately begin final at the calibration target: raw-state overlap must be reported,
    # not hidden by a start-only audit.
    final_start = tuple(calibration[0]["target"])
    final = [_atomic_row("final", final_start, "SC2")]
    raw = {"calibration": calibration, "final": final}
    manifest = {f"{split}.jsonl": core.digest_jsonl_rows(rows) for split, rows in raw.items()}
    reference = core.audit_atomic_reference(manifest, raw)
    assert reference.ok
    assert reference.report["state_overlap_between_atomic_splits"]["calibration:final"] >= 1
    assert len(reference.forbidden_state_fingerprints) >= 3
    assert len(reference.report["all_raw_state_fingerprint_sha256"]) == 64

    bad = copy.deepcopy(manifest)
    bad["final.jsonl"]["sha256"] = "0" * 64
    with pytest.raises(core.CompositionIntegrityError, match="digest mismatch"):
        core.audit_atomic_reference(bad, raw)


def test_small_generation_is_deterministic_exact_clean_and_disjoint() -> None:
    reference = _atomic_reference()
    registry_one = core.FingerprintRegistry.from_atomic_reference(reference)
    dev_a = core.generate_split(
        "dev_a",
        12,
        "A",
        seed=731,
        depths=(2, 3, 4),
        depth_counts={2: 4, 3: 4, 4: 4},
        registry=registry_one,
    )
    registry_two = core.FingerprintRegistry.from_atomic_reference(reference)
    repeated = core.generate_split(
        "dev_a",
        12,
        "A",
        seed=731,
        depths=(2, 3, 4),
        depth_counts={2: 4, 3: 4, 4: 4},
        registry=registry_two,
    )
    assert repeated == dev_a
    assert Counter(row["depth"] for row in dev_a) == {2: 4, 3: 4, 4: 4}
    assert all(row["shortest_depth"] == row["depth"] for row in dev_a)
    assert all(row["motif_count"] == 0 for row in dev_a)
    assert all(len(row["solutions"]) <= 2 for row in dev_a)

    dev_b = core.generate_split("dev_b", 10, "B", seed=732, depths=(3,), registry=registry_one)
    assert all(row["motif_count"] == 1 for row in dev_b)
    assert all(
        core.motif_count(solution["program"]) == 1
        for row in dev_b
        for solution in row["solutions"]
    )
    audit = core.audit_splits({"dev_a": dev_a, "dev_b": dev_b}, atomic_reference=reference)
    assert audit["ok"], audit["errors"]
    assert not any(audit["state_overlap"].values())
    assert not any(audit["atomic_state_overlap"].values())


def test_split_audit_detects_semantic_duplicates_and_atomic_raw_state_overlap() -> None:
    row = core.generate_split("dev_a", 1, "A", seed=900, depths=(2,))[0]
    duplicate = copy.deepcopy(row)
    duplicate["split"] = "shadow"
    local = core.audit_splits({"dev_a": [row], "shadow": [duplicate]})
    assert not local["ok"]
    assert local["state_overlap"]["dev_a:shadow"] > 0
    assert local["task_overlap"]["dev_a:shadow"] == 1

    same_start = tuple(row["start"])
    raw = {
        "calibration": [_atomic_row("calibration", same_start, "AC1", p=int(row["p"]))],
        "final": [_atomic_row("final", (1, 4, 7, 10), "SC2")],
    }
    manifest = {f"{split}.jsonl": core.digest_jsonl_rows(rows) for split, rows in raw.items()}
    reference = core.audit_atomic_reference(manifest, raw)
    cross = core.audit_splits({"dev_a": [row]}, atomic_reference=reference)
    assert not cross["ok"]
    assert cross["atomic_state_overlap"]["dev_a"] >= 1


def test_discover_balancing_is_deterministic_unique_and_covers_every_position() -> None:
    rows = []
    index = 0
    for depth in (2, 3, 4):
        for op in core.OPS:
            rows.append(_synthetic_task(index, depth, op))
            index += 1
    thresholds = core.DiscoveryThresholds(
        min_unique_trajectories=15,
        min_solvable_tasks=15,
        min_full_signatures=15,
        min_position_count=1,
    )
    selected, report = core.exact_discover(rows, target_count=15, thresholds=thresholds)
    repeated, repeated_report = core.exact_discover(list(reversed(rows)), target_count=15, thresholds=thresholds)
    assert [row["trajectory_fingerprint"] for row in selected] == [row["trajectory_fingerprint"] for row in repeated]
    assert report == repeated_report
    assert report["passes_minimums"]
    assert report["unique_trajectories"] == 15
    assert report["solvable_tasks"] == 15
    assert report["first_operation_spread"] == 0
    assert report["checks"]["first_operation_balance"]
    assert report["minimum_position_count"] == 1
    assert all(row["motif_count"] == 0 for row in selected)


def test_discover_gate_rejects_missing_first_operation_pool() -> None:
    rows = [_synthetic_task(index, 2, op) for index, op in enumerate(core.OPS[:-1])]
    thresholds = core.DiscoveryThresholds(min_unique_trajectories=4, min_solvable_tasks=4,
                                          min_full_signatures=4, min_position_count=0,
                                          max_first_operation_spread=1)
    _, report = core.exact_discover(rows, target_count=4, thresholds=thresholds)
    assert report["checks"]["first_operation_balance"] is False
    assert report["passes_minimums"] is False


def test_discover_balancer_meets_exact_first_op_quota_despite_task_conflict(monkeypatch) -> None:
    candidates = []
    for depth in (2, 3, 4):
        for op_index, op in enumerate(core.OPS):
            # The first AX1 candidate conflicts with the first SH1 candidate.
            # A simple depth/pool round-robin therefore skips AX1 once and used
            # to stop at a 3/2/2/2/1 split for target_count=10.
            task_id = "shared-depth2" if depth == 2 and op in ("SH1", "AX1") else f"t-{depth}-{op}"
            candidates.append({
                "trajectory_id": f"trajectory-{depth}-{op}",
                "trajectory_fingerprint": f"{depth:02d}-{op_index:02d}-{op}",
                "task_id": task_id,
                "depth": depth,
                "program": [op] * depth,
                "motif_count": 0,
            })
    monkeypatch.setattr(core, "_discovery_candidates", lambda _rows: candidates)
    thresholds = core.DiscoveryThresholds(min_unique_trajectories=10, min_solvable_tasks=10,
                                           min_full_signatures=5, min_position_count=1,
                                           max_first_operation_spread=0)
    selected, report = core.balance_discovered_trajectories([], target_count=10, thresholds=thresholds)
    assert len(selected) == 10
    assert report["first_operation"] == {op: 2 for op in sorted(core.OPS)}
    assert report["first_operation_target"] == {op: 2 for op in sorted(core.OPS)}
    assert report["first_operation_spread"] == 0
    assert report["checks"]["first_operation_balance"] is True


def test_scientific_size_and_final_freeze_guards_fail_before_generation() -> None:
    reference = _atomic_reference()
    with pytest.raises(ValueError, match="at least 4000"):
        core.generate_pilot_data(
            reference,
            train_depth_counts={2: 100, 3: 100, 4: 100},
            dev_a_count=500,
            dev_b_count=500,
        )
    with pytest.raises(ValueError, match="at least 500"):
        core.generate_pilot_data(
            reference,
            train_depth_counts={2: 1000, 3: 2000, 4: 1000},
            dev_a_count=499,
            dev_b_count=500,
        )
    with pytest.raises(core.CompositionIntegrityError, match="locked"):
        core.generate_final_data(reference, {}, config_frozen=False)


def test_confirm_atomic_data_is_fresh_exact_and_balanced() -> None:
    reference = _atomic_reference()
    registry = core.FingerprintRegistry.from_atomic_reference(reference)
    pilot = {"dev_a": core.generate_split("dev_a", 3, "A", seed=1201, depths=(2,), registry=registry)}
    final = {"final_c": core.generate_split("final_c", 3, "C", seed=1202, depths=(2,), registry=registry)}
    rows = core.generate_confirm_atomic_data(reference, pilot, final, per_operation=3, seed=1203)
    assert len(rows) == 15
    assert Counter(row["operation"] for row in rows) == Counter({op: 3 for op in core.OPS})
    for op in core.OPS:
        assert Counter(row["degree"] for row in rows if row["operation"] == op) == {2: 1, 3: 1, 4: 1}
    assert all(row["depth"] == 1 and row["shortest_depth"] == 1 for row in rows)
    assert all(core.verify_program(row["start"], row["target"], row["witness"], row["p"]) for row in rows)
    confirm_states = set().union(*(core._row_state_fingerprints(row) for row in rows))
    composition_states = set().union(
        *(core._row_state_fingerprints(row) for split in (pilot, final) for group in split.values() for row in group)
    )
    assert not (confirm_states & reference.forbidden_state_fingerprints)
    assert not (confirm_states & composition_states)
    assert core.audit_splits({"confirm_atomic": rows}, atomic_reference=reference)["ok"]


def test_public_integration_api_is_present() -> None:
    expected = {
        "OPS",
        "TOKENS",
        "HELD_MOTIFS",
        "apply_op",
        "trajectory",
        "enumerate_programs",
        "format_state",
        "plan_prompt",
        "apply_prompt",
        "program_answer",
        "task_fingerprint",
        "state_fingerprint",
        "generate_pilot_data",
        "exact_discover",
        "audit_pilot_data",
        "generate_final_data",
        "generate_confirm_atomic_data",
    }
    assert expected <= set(core.__all__)

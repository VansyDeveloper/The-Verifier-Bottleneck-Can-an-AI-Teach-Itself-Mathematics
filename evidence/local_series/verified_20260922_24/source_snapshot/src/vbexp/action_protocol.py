from __future__ import annotations

OPERATIONS = ("SH1", "SC2", "REV", "AC1", "AX1")


def forced_first_action(method: str, candidate_id: int, k: int):
    if method != "prefix_balanced_action":
        return None
    forced_count = max(k - 2, 0)
    return OPERATIONS[candidate_id % len(OPERATIONS)] if candidate_id < forced_count else None


def forced_training_first(method: str, candidate_id: int):
    if method == "prefix_balanced_action" and candidate_id < len(OPERATIONS):
        return OPERATIONS[candidate_id]
    return None


def unique_program_count(programs):
    return len(set(programs))

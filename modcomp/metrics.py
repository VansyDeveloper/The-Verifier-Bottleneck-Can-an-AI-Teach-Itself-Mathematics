"""Small, model-agnostic metrics shared by training and analysis."""

from math import log2


def summarize_checker_counts(counts):
    """Summarize a verifier confusion table collected on training rollouts."""
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    groups = int(counts.get("groups", 0))
    zero_variance_groups = int(counts.get("zero_variance_groups", 0))
    if min(tp, fp, fn, tn, groups, zero_variance_groups) < 0:
        raise ValueError("checker counts must be non-negative")
    if zero_variance_groups > groups:
        raise ValueError("zero-variance groups cannot exceed all groups")

    total = tp + fp + fn + tn
    correct = tp + fn
    wrong = fp + tn
    accepted = tp + fp

    mutual_information = 0.0
    if total:
        cells = (
            (tp, correct, accepted),
            (fn, correct, total - accepted),
            (fp, wrong, accepted),
            (tn, wrong, total - accepted),
        )
        for joint_count, correctness_count, verdict_count in cells:
            if joint_count:
                p_joint = joint_count / total
                mutual_information += p_joint * log2(
                    joint_count * total / (correctness_count * verdict_count)
                )

    tpr = tp / correct if correct else None
    fpr = fp / wrong if wrong else None
    result = {
        "n_training_verdicts": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "realized_tpr": tpr,
        "realized_fpr": fpr,
        "realized_signed_alignment": (
            tpr - fpr if tpr is not None and fpr is not None else None
        ),
        "acceptance_rate": accepted / total if total else None,
        "mutual_information_bits": mutual_information if total else None,
    }
    if "groups" in counts or "zero_variance_groups" in counts:
        result.update(
            {
                "groups": groups,
                "zero_variance_groups": zero_variance_groups,
                "zero_variance_group_rate": (
                    zero_variance_groups / groups if groups else None
                ),
            }
        )
    return result


def base_solved_summary(selection, baseline, final):
    """Measure change on problems selected by an independent baseline batch."""
    selection = {int(key): value for key, value in selection.items()}
    baseline = {int(key): value for key, value in baseline.items()}
    final = {int(key): value for key, value in final.items()}
    if (
        not selection
        or selection.keys() != baseline.keys()
        or baseline.keys() != final.keys()
    ):
        raise ValueError("selection, baseline, and final must contain the same problem set")

    per_problem = []
    for problem_index in sorted(baseline):
        selected_with = selection[problem_index]
        before = baseline[problem_index]
        after = final[problem_index]
        if selected_with["n"] <= 0 or before["n"] <= 0 or after["n"] <= 0:
            raise ValueError("sample counts must be positive")
        if selected_with["n"] != before["n"] or before["n"] != after["n"]:
            raise ValueError("selection, baseline, and final must use the same candidate count")
        valid_selection = 0 <= selected_with["correct"] <= selected_with["n"]
        valid_before = 0 <= before["correct"] <= before["n"]
        valid_after = 0 <= after["correct"] <= after["n"]
        if not valid_selection or not valid_before or not valid_after:
            raise ValueError("correct counts must lie between zero and sample count")
        base_solved = selected_with["correct"] > 0
        base_measurement_solved = before["correct"] > 0
        final_solved = after["correct"] > 0
        per_problem.append(
            {
                "problem_index": problem_index,
                "selection_correct": selected_with["correct"],
                "selection_n": selected_with["n"],
                "base_correct": before["correct"],
                "base_n": before["n"],
                "final_correct": after["correct"],
                "final_n": after["n"],
                "base_solved": base_solved,
                "base_measurement_solved": base_measurement_solved,
                "final_solved": final_solved,
                "not_rediscovered": base_solved and not final_solved,
            }
        )

    base_solved_rows = [row for row in per_problem if row["base_solved"]]
    not_rediscovered = sum(row["not_rediscovered"] for row in base_solved_rows)

    def pass1(rows, prefix):
        return sum(row[f"{prefix}_correct"] for row in rows) / sum(
            row[f"{prefix}_n"] for row in rows
        )

    baseline_subset_pass1 = (
        pass1(base_solved_rows, "base") if base_solved_rows else None
    )
    final_subset_pass1 = (
        pass1(base_solved_rows, "final") if base_solved_rows else None
    )
    return {
        "n_problems": len(per_problem),
        "base_pass@1": pass1(per_problem, "base"),
        "final_pass@1": pass1(per_problem, "final"),
        "selection_pass@k": sum(row["base_solved"] for row in per_problem)
        / len(per_problem),
        "base_pass@k": sum(row["base_measurement_solved"] for row in per_problem)
        / len(per_problem),
        "final_pass@k": sum(row["final_solved"] for row in per_problem) / len(per_problem),
        "base_solved_count": len(base_solved_rows),
        "base_solved_nonrediscovery_count": not_rediscovered,
        "base_solved_nonrediscovery_rate": (
            not_rediscovered / len(base_solved_rows) if base_solved_rows else None
        ),
        "base_pass@1_on_base_solved": baseline_subset_pass1,
        "final_pass@1_on_base_solved": final_subset_pass1,
        "pass@1_change_on_base_solved": (
            final_subset_pass1 - baseline_subset_pass1
            if base_solved_rows
            else None
        ),
        "per_problem": per_problem,
    }

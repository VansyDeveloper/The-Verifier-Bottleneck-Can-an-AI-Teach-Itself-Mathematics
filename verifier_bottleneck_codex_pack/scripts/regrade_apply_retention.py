"""Mode-aware re-grading of the stored D-021/D-022 APPLY retention generations.

The registered retention check grades an APPLY answer with the APPLY parser,
which accepts only `RESULT: [c0, ..., cd]`.  Every composition adapter answers
`PROGRAM: ...` instead, so the registered numbers (pass@1 0.00, parse rate 0.00)
merge two different failures that carry different scientific meaning:

  * response-format drift  - the adapter answers a *different question* than the
    one it was asked, because `--apply-weight 0` removed every APPLY example from
    its 1875-step training stream (`train_sft.py`: tasks = apply * weight + plan);
  * capability loss        - the adapter can no longer do the modular arithmetic.

This script separates them on generations that are already on disk.  It makes no
model call, needs no GPU, and does not touch any existing report.  For an answer
in PLAN format it additionally asks whether the emitted program is a correct
plan for the same (start, target) pair, i.e. whether the adapter competently
solved the PLAN version of the APPLY prompt it was given.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from vbexp.io import read_tasks
from vbexp.polynomial import apply_program
from vbexp.verifier import verify

ROLES = ["atomic", "oracle0", "oracle1", "oracle2", "nomotif0", "nomotif1", "nomotif2"]
APPLY_SPLIT = "sft_validation_apply_non_sh1"


def role_of(adapter: str) -> str | None:
    adapter = adapter.replace("\\", "/")
    if adapter.endswith("sft_atomic_r32_pilot_aw4_cont"):
        return "atomic"
    if "composition_oracle_seed" in adapter:
        return f"oracle{adapter[-1]}"
    if "composition_nomotif_seed" in adapter:
        return f"nomotif{adapter[-1]}"
    return None


def discover(runs_root: Path) -> dict[str, Path]:
    """One complete GPU retention run per role on the APPLY split."""
    found: dict[str, Path] = {}
    for run in sorted(runs_root.iterdir()):
        config_path = run / "config.resolved.yaml"
        if not (run / "DONE").exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            continue
        if not str(config.get("label", "")).startswith("composition-forget-"):
            continue
        if Path(str(config.get("input", "")).replace("\\", "/")).stem != APPLY_SPLIT:
            continue
        environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        if environment.get("torch", {}).get("cuda_available") is not True:
            continue
        role = role_of(str(config.get("adapter", "")))
        if role is None:
            continue
        if role in found:
            raise RuntimeError(f"two complete GPU APPLY retention runs for {role}")
        found[role] = run
    return found


_LINE = re.compile(r"^\s*(RESULT|PROGRAM)\s*:\s*(.*?)\s*$", re.IGNORECASE | re.DOTALL)


def grade(run: Path, tasks: dict[str, object]) -> dict:
    """Classify each answer by *what it contains*, independently of its prefix.

    An APPLY prompt asks for a coefficient vector. These adapters answer with an
    operation sequence, under either the `RESULT:` or the `PROGRAM:` prefix, so
    prefix alone does not identify the failure. The content decides.
    """
    counts = {
        "tasks": 0,
        "registered_pass_at_1": 0,
        "prefix_result": 0,
        "prefix_program": 0,
        "prefix_missing": 0,
        "content_state_vector": 0,
        "content_state_vector_correct": 0,
        "content_program": 0,
        "content_program_is_a_correct_plan": 0,
        "content_program_equals_prompt_program": 0,
        "content_other": 0,
    }
    examples: list[dict] = []
    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task = tasks[record["task_id"]]
        text = record["text"].strip()
        counts["tasks"] += 1
        if record.get("is_correct"):
            counts["registered_pass_at_1"] += 1

        match = _LINE.match(text)
        if match is None:
            counts["prefix_missing"] += 1
            counts["content_other"] += 1
            continue
        prefix, payload = match.group(1).upper(), match.group(2).strip()
        counts["prefix_result" if prefix == "RESULT" else "prefix_program"] += 1

        if payload.startswith("["):
            counts["content_state_vector"] += 1
            if verify(task, f"RESULT: {payload}").is_correct:
                counts["content_state_vector_correct"] += 1
            continue

        tokens = payload.split()
        if tokens and all(op in task.operations for op in tokens):
            counts["content_program"] += 1
            solves = apply_program(task.start, tuple(tokens), task.p) == task.target
            if solves:
                counts["content_program_is_a_correct_plan"] += 1
            if task.program is not None and tuple(tokens) == tuple(task.program):
                counts["content_program_equals_prompt_program"] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "task_id": record["task_id"],
                        "prompt_program": list(task.program or []),
                        "answer": text,
                        "is_a_correct_plan": solves,
                    }
                )
            continue
        counts["content_other"] += 1

    n = max(counts["tasks"], 1)
    return {
        **counts,
        "content_program_rate": counts["content_program"] / n,
        "registered_pass_at_1_rate": counts["registered_pass_at_1"] / n,
        "correct_plan_rate": counts["content_program_is_a_correct_plan"] / n,
        "examples": examples,
    }


def report(payload: dict) -> str:
    rows = payload["by_role"]
    lines = [
        "# Mode-aware re-grading of the APPLY retention check",
        "",
        "The registered check accepts only `RESULT: [c0, ..., cd]`. This table re-reads the",
        "same stored generations and classifies each answer by **what it contains** rather",
        "than by its prefix, because these adapters emit an operation sequence under both",
        "the `RESULT:` and the `PROGRAM:` prefix. No model was run; no registered report",
        "was modified.",
        "",
        "| adapter | n | registered pass@1 | content: state vector | of those correct | content: a program | of those a correct plan | content: other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        item = rows.get(role)
        if item is None:
            lines.append(f"| {role} | missing | | | | | | |")
            continue
        lines.append(
            f"| {role} | {item['tasks']} | {item['registered_pass_at_1']} | "
            f"{item['content_state_vector']} | {item['content_state_vector_correct']} | "
            f"{item['content_program']} | {item['content_program_is_a_correct_plan']} | "
            f"{item['content_other']} |"
        )
    trained = [rows[r] for r in ROLES if r != "atomic" and r in rows]
    if trained:
        n = sum(item["tasks"] for item in trained)
        vec = sum(i["content_state_vector"] for i in trained)
        prog = sum(i["content_program"] for i in trained)
        plan_ok = sum(i["content_program_is_a_correct_plan"] for i in trained)
        pref_result = sum(i["prefix_result"] for i in trained)
        lines += [
            "",
            "## Reading",
            "",
            f"Across the {len(trained)} composition adapters ({n} graded answers):",
            f"- {vec}/{n} answers actually contain a coefficient vector, which is what an APPLY",
            "  prompt asks for.",
            f"- {prog}/{n} contain an operation sequence instead - the answer to the PLAN version",
            f"  of the same task. Of those, {plan_ok} are a *correct plan* for the same",
            "  (start, target) pair, so the output is competent for a question that was not asked.",
            f"- {pref_result}/{n} still carry the correct `RESULT:` prefix while containing a",
            "  program, so this is a collapse of the answer *content*, not merely of the wrapper.",
            "",
            "The registered `pass@1 = 0.00, parse rate = 0.00` therefore measures a collapse of",
            "mode-conditional response behaviour, not destruction of modular arithmetic. The direct",
            "cause is recorded in the drivers: `run_capacity_series.sh` and `run_amended_series.sh`",
            "pass `--apply-weight 0`, and `train_sft.py` builds the stream as",
            "`apply_tasks * apply_weight + plan_tasks`, so the 1875-step composition run saw zero",
            "APPLY targets. The 20% replay declared in the capacity preregistration was PLAN-only.",
            "",
            "This does not cancel the retention concern - a policy that cannot be addressed in APPLY",
            "mode is not a usable multi-mode policy - but what these data license is *mode collapse",
            "caused by removing APPLY from the training mixture*, not catastrophic forgetting of a",
            "skill. The `applykeep` arm of the night series tests the causal claim directly.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/pilot"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/apply_retention_regrade.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/reports/APPLY_RETENTION_REGRADE.md")
    )
    args = parser.parse_args()

    tasks = {task.task_id: task for task in read_tasks(args.data / f"{APPLY_SPLIT}.jsonl")}
    found = discover(args.runs)
    by_role = {role: grade(run, tasks) for role, run in sorted(found.items())}
    payload = {
        "purpose": "separate response-format drift from capability loss in the APPLY retention check",
        "split": APPLY_SPLIT,
        "grader": "mode-aware: RESULT parsed as APPLY, PROGRAM re-checked as a PLAN answer",
        "runs": {role: run.name for role, run in sorted(found.items())},
        "missing_roles": [role for role in ROLES if role not in found],
        "by_role": by_role,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(report(payload), encoding="utf-8")

    for role in ROLES:
        item = by_role.get(role)
        if item is None:
            print(f"{role}: MISSING")
            continue
        print(
            f"{role}: n={item['tasks']} vector={item['content_state_vector']} "
            f"(correct {item['content_state_vector_correct']}) "
            f"program={item['content_program']} "
            f"(a correct plan {item['content_program_is_a_correct_plan']}) "
            f"other={item['content_other']} registered_pass@1={item['registered_pass_at_1']}"
        )
    print(f"\nwritten: {args.output}")
    print(f"written: {args.report}")


if __name__ == "__main__":
    main()

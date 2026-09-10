from __future__ import annotations

from .task import Task


def format_state(state: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(v) for v in state) + "]"


def build_prompt(task: Task) -> str:
    if task.mode == "apply":
        if task.program is None:
            raise ValueError("apply task requires program")
        return (
            f"FIELD: {task.p}\n"
            f"DEGREE_CAP: {task.degree_cap}\n"
            f"START: {format_state(task.start)}\n"
            f"PROGRAM: {' '.join(task.program)}\n"
            "Return exactly one line:\n"
            "RESULT: [c0, c1, ..., cd]"
        )
    if task.mode == "plan":
        if task.target is None or task.max_steps is None:
            raise ValueError("plan task requires target and max_steps")
        return (
            f"FIELD: {task.p}\n"
            f"DEGREE_CAP: {task.degree_cap}\n"
            f"START: {format_state(task.start)}\n"
            f"TARGET: {format_state(task.target)}\n"
            f"ALLOWED: {' '.join(task.operations)}\n"
            f"MAX_STEPS: {task.max_steps}\n"
            "Return exactly one line:\n"
            "PROGRAM: OP1 OP2 ..."
        )
    raise ValueError(f"unsupported mode: {task.mode}")

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .polynomial import apply_program
from .task import Task

_RESULT_RE = re.compile(r"^\s*RESULT\s*:\s*\[([^\]]*)\]\s*$", re.IGNORECASE)
_PROGRAM_RE = re.compile(r"^\s*PROGRAM\s*:\s*(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class VerificationResult:
    parse_ok: bool
    is_correct: bool
    parsed_result: tuple[int, ...] | None = None
    parsed_program: tuple[str, ...] | None = None
    error: str | None = None


def parse_result(text: str) -> tuple[int, ...]:
    match = _RESULT_RE.match(text.strip())
    if not match:
        raise ValueError("expected exactly: RESULT: [..]")
    payload = match.group(1).strip()
    if not payload:
        return tuple()
    return tuple(int(part.strip()) for part in payload.split(","))


def parse_program(text: str) -> tuple[str, ...]:
    match = _PROGRAM_RE.match(text.strip())
    if not match:
        raise ValueError("expected exactly: PROGRAM: OP1 OP2 ...")
    payload = match.group(1).strip()
    return tuple(payload.split()) if payload else tuple()


def verify(task: Task, text: str) -> VerificationResult:
    try:
        if task.mode == "apply":
            if task.target is None:
                raise ValueError("apply task is missing target")
            parsed = parse_result(text)
            normalized = tuple(value % task.p for value in parsed)
            correct = len(normalized) == task.degree_cap + 1 and normalized == task.target
            return VerificationResult(True, correct, parsed_result=normalized)

        if task.mode == "plan":
            if task.target is None or task.max_steps is None:
                raise ValueError("plan task is missing target/max_steps")
            program = parse_program(text)
            if len(program) > task.max_steps:
                return VerificationResult(True, False, parsed_program=program, error="program_too_long")
            if any(op not in task.operations for op in program):
                return VerificationResult(True, False, parsed_program=program, error="unknown_operation")
            result = apply_program(task.start, program, task.p)
            return VerificationResult(True, result == task.target, parsed_program=program)

        raise ValueError(f"unsupported mode: {task.mode}")
    except Exception as exc:
        return VerificationResult(False, False, error=f"{type(exc).__name__}: {exc}")

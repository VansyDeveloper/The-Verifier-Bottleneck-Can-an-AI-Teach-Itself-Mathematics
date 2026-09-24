from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskMode = Literal["apply", "plan"]


@dataclass(frozen=True)
class Task:
    task_id: str
    mode: TaskMode
    split: str
    p: int
    degree_cap: int
    start: tuple[int, ...]
    operations: tuple[str, ...]
    target: tuple[int, ...] | None = None
    program: tuple[str, ...] | None = None
    max_steps: int | None = None
    difficulty: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("start", "operations", "target", "program"):
            if value.get(key) is not None:
                value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        data = dict(value)
        for key in ("start", "operations", "target", "program"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .task import Task


def write_tasks(path: str | Path, tasks: Iterable[Task]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def read_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks

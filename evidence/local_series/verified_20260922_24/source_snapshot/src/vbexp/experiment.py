from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def environment_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
    }
    try:
        import importlib.metadata

        result["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "PyYAML")
        }
    except Exception as exc:
        result["package_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    return result


def git_state() -> str:
    commands = (["git", "rev-parse", "HEAD"], ["git", "status", "--short", "--branch"])
    parts: list[str] = []
    for command in commands:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        parts.append(f"$ {' '.join(command)}\n{completed.stdout}{completed.stderr}")
    return "\n".join(parts)


@dataclass
class RunDirectory:
    run_id: str
    path: Path
    config: dict[str, Any]
    seed: int
    started_at: str

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        model_tag: str,
        task: str,
        method: str,
        verifier: str,
        seed: int,
        config: dict[str, Any],
        output_root: str | Path = "artifacts/runs",
    ) -> "RunDirectory":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}_{model_tag}_{task}_{kind}_{method}_{verifier}_seed{seed}"
        path = Path(output_root) / run_id
        if path.exists() and (path / "DONE").exists():
            raise FileExistsError(f"completed run already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoints").mkdir(exist_ok=True)
        yaml_text = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
        (path / "config.resolved.yaml").write_text(yaml_text, encoding="utf-8")
        write_json(path / "environment.json", environment_snapshot())
        (path / "git_state.txt").write_text(git_state(), encoding="utf-8")
        (path / "stdout.log").touch()
        (path / "stderr.log").touch()
        (path / "metrics.jsonl").touch()
        (path / "generations.jsonl").touch()
        started_at = utc_now()
        write_json(
            path / "run_manifest.json",
            {
                "run_id": run_id,
                "status": "RUNNING",
                "config_sha256": hashlib.sha256(yaml_text.encode("utf-8")).hexdigest(),
                "seed": seed,
                "started_at": started_at,
                "git_commit": None,
                "model": {"name_or_path": config.get("model_name_or_path")},
                "environment": environment_snapshot(),
                "artifacts": [],
            },
        )
        return cls(run_id, path, config, seed, started_at)

    def log(self, text: str, *, stderr: bool = False) -> None:
        target = self.path / ("stderr.log" if stderr else "stdout.log")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")

    def metric(self, split: str, step: int, metrics: dict[str, float | int | None]) -> None:
        append_jsonl(
            self.path / "metrics.jsonl",
            [{"run_id": self.run_id, "split": split, "step": step, "metrics": metrics}],
        )

    def generations(self, records: Iterable[dict[str, Any]]) -> None:
        append_jsonl(self.path / "generations.jsonl", records)

    def finish(self, metrics: dict[str, Any]) -> None:
        write_json(self.path / "final_metrics.json", metrics)
        (self.path / "DONE").write_text(utc_now() + "\n", encoding="utf-8")
        manifest = json.loads((self.path / "run_manifest.json").read_text(encoding="utf-8"))
        manifest.update(status="DONE", finished_at=utc_now())
        write_json(self.path / "run_manifest.json", manifest)

    def fail(self, exc: BaseException) -> None:
        import traceback

        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.log(trace, stderr=True)
        (self.path / "FAILED").write_text(utc_now() + "\n", encoding="utf-8")
        manifest = json.loads((self.path / "run_manifest.json").read_text(encoding="utf-8"))
        manifest.update(status="FAILED", finished_at=utc_now())
        write_json(self.path / "run_manifest.json", manifest)


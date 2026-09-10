from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(command: list[str]):
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
        return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "disk": shutil.disk_usage(Path.cwd())._asdict(),
        "packages": {name: package_version(name) for name in [
            "torch", "transformers", "datasets", "accelerate", "peft", "trl", "numpy", "pandas", "scipy"
        ]},
        "nvidia_smi": run_command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
        "git_status": run_command(["git", "status", "--short"]),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
    }

    try:
        import torch
        report["torch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
    except Exception as exc:
        report["torch_runtime"] = {"error": f"{type(exc).__name__}: {exc}"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

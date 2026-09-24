"""RTX 3060 entrypoint for the prospective hardware amendment.

The frozen V100 trainer is reused byte-for-byte. Only its process-local GPU
guard is replaced, and the resolved config receives an RTX-specific schema.
The original runner and frozen SHA-256 manifest are never changed.
"""

from __future__ import annotations

import json
import os
import subprocess


def parse_visible_uuid(output: str) -> str:
    uuids = [line.strip() for line in output.splitlines() if line.strip()]
    if len(uuids) != 1 or not uuids[0].startswith("GPU-"):
        raise ValueError("exactly one visible GPU UUID is required")
    return uuids[0]


def validate_device_binding(name: str, actual_uuid: str, assigned_uuid: str) -> None:
    if "RTX 3060" not in name:
        raise ValueError(f"amended FP16 runner requires RTX 3060, got {name}")
    if not assigned_uuid or actual_uuid != assigned_uuid:
        raise ValueError("GPU UUID differs from prospective assignment")


def amended_fp16_runtime(resolved: dict) -> dict:
    import torch

    if resolved.get("dtype") != "float16" or not torch.cuda.is_available():
        raise RuntimeError("amended runner requires CUDA FP16")
    name = torch.cuda.get_device_name(torch.cuda.current_device())
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    uuid = parse_visible_uuid(result.stdout)
    validate_device_binding(name, uuid, os.environ.get("ASSIGNED_GPU_UUID", ""))
    return {
        "runtime_device": "cuda", "runtime_dtype": "torch.float16",
        "runtime_compute_dtype": "torch.float16", "runtime_master_dtype": "torch.float32",
        "runtime_autocast_enabled": True, "cuda_device_name": name,
        "runtime_gpu_uuid": uuid, "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def main() -> None:
    import fp16_training
    import run_first

    fp16_training.fp16_runtime = amended_fp16_runtime
    run_first.fp16_runtime = amended_fp16_runtime
    original_prepared_pair = run_first.prepared_pair
    original_accept_pair = run_first.accept_pair

    def prepared_pair(*args, **kwargs):
        prepared, resolved = original_prepared_pair(*args, **kwargs)
        return prepared, {**resolved, "schema": "withheld-pairs.rtx3060-fp16-resolved.v1"}

    def accept_pair(freeze, entry, seed, model_lib):
        assigned = os.environ.get("ASSIGNED_GPU_UUID", "")
        output = run_first.OUTPUT
        for arm in ("random", "withheld"):
            path = output / "adapters" / f"k{entry['k']}_s{entry['subset']}_seed{seed}_{arm}" / "training_receipt.json"
            if path.exists():
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if receipt.get("runtime_gpu_uuid") != assigned:
                    raise RuntimeError("paired arm GPU UUID differs from assignment")
        return original_accept_pair(freeze, entry, seed, model_lib)

    run_first.prepared_pair = prepared_pair
    run_first.accept_pair = accept_pair
    run_first.main()


if __name__ == "__main__":
    main()

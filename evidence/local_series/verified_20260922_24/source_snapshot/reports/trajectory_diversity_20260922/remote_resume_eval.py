"""Complete the interrupted held-out evaluations on one NetBird V100.

The script reads the frozen model and adapters. It never trains a model.
Seeds 83004 and 83005 are both evaluated in full so each paired comparison
uses one GPU and one numerical implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT.parent
CODE = ROOT / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code"
sys.path.insert(0, str(CODE))
# The legacy module imports v5_train eagerly. Its only references are in
# training/export functions and the tokenizer branch without an export path.
# Evaluation uses the frozen local export and does not call those functions.
sys.modules["v5_train"] = types.ModuleType("v5_train")

import torch

import composition_eval as eval_lib
import composition_model as model_lib


EXPECTED_FREEZE = "609EA5657269B4371A3033DE29BC2099CF629549F3133A3C8F3DC32E492FE9D0"
EXPECTED_ATOMIC_DATA = "DD084D13159D545F85F4FC89A3A47898D22A853302A2481D7301B45AC4E3373F"
INPUT = STUDY
EXPORT = INPUT / "export"
ADAPTERS = INPUT / "adapters"
DATA = INPUT / "data"
OUTPUT = INPUT / "results/rankings"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def windows_tree_sha256(path: Path) -> str:
    """Reproduce the frozen WindowsPath ordering for adapter tree receipts."""
    digest = hashlib.sha256()
    files = sorted((item for item in path.rglob("*") if item.is_file()),
                   key=lambda item: item.relative_to(path).as_posix().casefold())
    if not files:
        raise RuntimeError(f"empty adapter tree: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_inputs() -> tuple[list[dict], list[dict], dict]:
    if not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("the assigned NetBird V100 is unavailable")
    freeze_path = DATA / "SELECTION_FROZEN.json"
    if sha256(freeze_path) != EXPECTED_FREEZE:
        raise RuntimeError("selection freeze SHA-256 mismatch")
    freeze = read_json(freeze_path)
    for relative, expected in freeze["code_sha256"].items():
        if sha256(ROOT / relative) != expected.upper():
            raise RuntimeError(f"frozen code SHA-256 mismatch: {relative}")
    holdout_path = DATA / "holdout_final_a.jsonl"
    holdout_receipt = read_json(DATA / "holdout_receipt.json")
    if sha256(holdout_path) != holdout_receipt["sha256"].upper():
        raise RuntimeError("holdout SHA-256 mismatch")
    atomic_path = DATA / "confirm_atomic.jsonl"
    if sha256(atomic_path) != EXPECTED_ATOMIC_DATA:
        raise RuntimeError("atomic reference data SHA-256 mismatch")
    rows = eval_lib.read_jsonl(holdout_path)
    atomic_rows = eval_lib.read_jsonl(atomic_path)
    if len(rows) != 1000 or len({row["task_id"] for row in rows}) != 1000:
        raise RuntimeError("holdout must contain 1000 unique tasks")
    for seed in (83004, 83005):
        pair = read_json(DATA / f"seed{seed}_pair.json")
        if pair.get("status") != "DONE" or pair["freeze_sha256"] != EXPECTED_FREEZE:
            raise RuntimeError(f"seed {seed}: training pair not accepted")
        for arm in ("random", "diverse"):
            adapter = ADAPTERS / f"seed{seed}_{arm}"
            if windows_tree_sha256(adapter) != pair["adapters"][arm]["tree_sha256"].upper():
                raise RuntimeError(f"seed {seed} {arm}: adapter tree SHA-256 mismatch")
            if sha256(adapter / "training_receipt.json") != pair["adapters"][arm]["receipt_sha256"].upper():
                raise RuntimeError(f"seed {seed} {arm}: training receipt SHA-256 mismatch")
    return rows, atomic_rows, holdout_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    rows, atomic_rows, holdout_receipt = verify_inputs()
    if args.check_only:
        print(json.dumps({"status": "INPUTS_VERIFIED", "gpu": torch.cuda.get_device_name(0),
                          "holdout_tasks": len(rows), "atomic_tasks": len(atomic_rows)}), flush=True)
        return
    branch_results = []
    for seed in (83004, 83005):
        pair = read_json(DATA / f"seed{seed}_pair.json")
        for arm in ("random", "diverse"):
            branch = f"seed{seed}_{arm}_holdout"
            adapter = ADAPTERS / f"seed{seed}_{arm}"
            binding = {
                "schema": "trajectory-diversity.eval-binding.v1",
                "seed": seed,
                "arm": arm,
                "holdout_sha256": holdout_receipt["sha256"],
                "adapter_tree_sha256": pair["adapters"][arm]["tree_sha256"],
                "selection_freeze_sha256": EXPECTED_FREEZE,
            }
            model, tokenizer, token_ids = model_lib.load_branch(EXPORT, adapter)
            try:
                if next(model.parameters()).dtype != torch.float32:
                    raise RuntimeError(f"{branch}: evaluation model is not FP32")
                summary, _ = eval_lib.evaluate_ranking(model, tokenizer, token_ids, rows, OUTPUT, branch, binding)
                atomic_binding = {**binding, "split": "confirm_atomic", "atomic_data_sha256": EXPECTED_ATOMIC_DATA}
                eval_lib.evaluate_atomic(model, tokenizer, token_ids, atomic_rows, OUTPUT,
                                         f"seed{seed}_{arm}_atomic", atomic_binding)
                result = {"seed": seed, "arm": arm, "hit32": summary["hit@32"], "atomic": "DONE"}
                branch_results.append(result)
                print(json.dumps({"milestone": "evaluation_done", **result}), flush=True)
            finally:
                eval_lib.unload(model)
    write_json(INPUT / "results/REMOTE_RESUME_DONE.json", {
        "schema": "trajectory-diversity.remote-resume.v1",
        "status": "DONE",
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "evaluation_dtype": "torch.float32",
        "freeze_sha256": EXPECTED_FREEZE,
        "holdout_sha256": holdout_receipt["sha256"],
        "branches": branch_results,
    })
    print(json.dumps({"status": "DONE", "branches": len(branch_results)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(INPUT / "results/REMOTE_RESUME_FAILED.json", {
            "schema": "trajectory-diversity.remote-resume.v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise

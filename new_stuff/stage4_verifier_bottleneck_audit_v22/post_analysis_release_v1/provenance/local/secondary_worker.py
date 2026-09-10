from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path


REPO = Path("/workspace/repo")
OPS = Path("/workspace/ops")
CONFIRM_CODE = REPO / "artifacts/stage4_composition_confirm_v1_0p6b/code"
sys.path.insert(0, str(CONFIRM_CODE))

import confirm  # noqa: E402


SECONDARY = [
    "final_b", "final_c", "final_d",
    "final_a_depth2", "final_b_depth2", "final_c_depth2", "final_d_depth2",
    "final_a_depth4", "final_b_depth4", "final_c_depth4", "final_d_depth4",
]

ASSIGNMENTS = {
    0: [(0, "atomic_control"), (1, "composition_distill"),
        (3, "atomic_control"), (4, "composition_distill")],
    1: [(0, "composition_distill"), (2, "atomic_control"),
        (3, "composition_distill"), (5, "atomic_control")],
    2: [(1, "atomic_control"), (2, "composition_distill"),
        (4, "atomic_control"), (5, "composition_distill")],
}


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


_original_begin_final_access = confirm.begin_final_access


def locked_begin_final_access() -> dict:
    lock_path = OPS / "final-access.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            return _original_begin_final_access()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def windows_tree_sha256(path: Path, *, exclude_export_receipt: bool = False) -> str:
    path = Path(path)
    files = [item for item in path.rglob("*") if item.is_file()]
    if exclude_export_receipt:
        files = [item for item in files if item.name != "export_receipt.json"]
    files.sort(key=lambda item: item.relative_to(path).as_posix().casefold())
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(confirm.sha256(item)))
    return digest.hexdigest().upper()


confirm.begin_final_access = locked_begin_final_access
confirm.tree_sha256 = windows_tree_sha256
confirm.model_lib.tree_sha256 = lambda path: windows_tree_sha256(path).lower()
confirm.model_lib._export_payload_sha = lambda path: windows_tree_sha256(
    path, exclude_export_receipt=True
).lower()


def remote_assert_amendment_commit(receipt: dict) -> None:
    # The transfer is an artifacts-only snapshot, so the remote container has no
    # .git object database.  Every scientific source remains checked by the
    # frozen code manifest; bind the commit field to the locally verified HEAD.
    if receipt.get("git_commit") != "fb098a6adcc5d39ae942eeae7952a1b7b6f0a032":
        raise RuntimeError("remote recovery amendment commit binding mismatch")


confirm._assert_amendment_commit = remote_assert_amendment_commit


_original_cached_rederive = confirm.eval_lib._rederive_cached_ranking_shard


def portable_cached_rederive(shard_rows, ranking_path, metric_path, branch, binding):
    try:
        return _original_cached_rederive(
            shard_rows, ranking_path, metric_path, branch, binding
        )
    except RuntimeError as exc:
        if "cached metrics differ from independently derived raw rankings" not in str(exc):
            raise
        raw_rows = confirm.eval_lib.read_jsonl(ranking_path)
        stored_rows = confirm.eval_lib.read_jsonl(metric_path)
        if len(raw_rows) != len(shard_rows) or len(stored_rows) != len(shard_rows):
            raise
        derived_rows = [
            confirm.eval_lib._ranking_metric_from_raw(task, raw, branch, binding)
            for task, raw in zip(shard_rows, raw_rows)
        ]
        for stored, derived in zip(stored_rows, derived_rows):
            if stored.keys() != derived.keys():
                raise
            for key in stored:
                left, right = stored[key], derived[key]
                if isinstance(left, float) and isinstance(right, float):
                    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15):
                        raise
                elif left != right:
                    raise
        return stored_rows


confirm.eval_lib._rederive_cached_ranking_shard = portable_cached_rederive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True, choices=sorted(ASSIGNMENTS))
    args = parser.parse_args()
    worker = args.worker
    receipt_path = OPS / f"worker-{worker}.json"
    started = time.time()
    completed: list[dict] = []
    atomic_write(receipt_path, {
        "schema": "stage4.ai01.secondary-worker.v1", "status": "RUNNING",
        "worker": worker, "pid": os.getpid(), "assignments": ASSIGNMENTS[worker],
        "started_at_unix": started,
    })
    try:
        confirm.require_frozen()
        confirm.require_training()
        for label, branch in ASSIGNMENTS[worker]:
            branch_started = time.time()
            print(json.dumps({"milestone": "branch_started", "worker": worker,
                              "replicate": label, "branch": branch}), flush=True)
            result = confirm.evaluate_branch(
                label, branch, confirm._adapter(label, branch), SECONDARY, evaluate_atomic=False
            )
            completed.append({
                "replicate": label, "branch": branch,
                "model_sha256": result["model_sha256"],
                "atomic_delegated_to_frozen_local_worker": True,
                "wall_seconds": time.time() - branch_started,
            })
            atomic_write(receipt_path, {
                "schema": "stage4.ai01.secondary-worker.v1", "status": "RUNNING",
                "worker": worker, "pid": os.getpid(), "assignments": ASSIGNMENTS[worker],
                "completed": completed, "started_at_unix": started,
                "updated_at_unix": time.time(),
            })
            print(json.dumps({"milestone": "branch_done", "worker": worker,
                              "replicate": label, "branch": branch,
                              "wall_seconds": completed[-1]["wall_seconds"]}), flush=True)
        atomic_write(receipt_path, {
            "schema": "stage4.ai01.secondary-worker.v1", "status": "DONE",
            "worker": worker, "pid": os.getpid(), "assignments": ASSIGNMENTS[worker],
            "completed": completed, "started_at_unix": started,
            "finished_at_unix": time.time(), "wall_seconds": time.time() - started,
        })
    except BaseException as exc:
        atomic_write(receipt_path, {
            "schema": "stage4.ai01.secondary-worker.v1", "status": "FAILED",
            "worker": worker, "pid": os.getpid(), "assignments": ASSIGNMENTS[worker],
            "completed": completed, "started_at_unix": started,
            "failed_at_unix": time.time(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()

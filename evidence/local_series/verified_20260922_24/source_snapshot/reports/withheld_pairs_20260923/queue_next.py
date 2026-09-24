"""Host-side queue for one prospectively assigned withheld-pair set.

The original model runner and frozen data are not edited. This coordinator
uses one Docker container per job, keeps every exited container, and refuses
to restart incomplete or failed evidence without a separate audit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import time
from pathlib import Path

from queue_first import FREEZE_SHA256, IMAGE, capture_container, command, log_event, sha256, write_once


SEEDS = (85000, 85001, 85002)
ARMS = ("random", "withheld")
CONTAINER_REPO = "/work/withheld_pairs_20260923/repo"


def ordered_jobs() -> list[tuple[str, int, str]]:
    return [(mode, seed, arm) for mode in ("train", "evaluate")
            for seed in SEEDS for arm in ARMS]


def resume_decision(accepted: bool, state: dict | None, actual_uuid: str | None,
                    assigned_uuid: str) -> str:
    if state is not None and actual_uuid != assigned_uuid:
        raise ValueError("container GPU UUID differs from assignment")
    if state is None:
        return "SKIP" if accepted else "START"
    if state.get("Status") == "running":
        return "WAIT"
    if state.get("Status") == "exited":
        if state.get("ExitCode") != 0:
            raise ValueError("prior container failed; preserve and audit its evidence")
        return "SKIP" if accepted else "VERIFY"
    raise ValueError(f"unexpected container state: {state.get('Status')}")


def gpu_is_available(snapshot: str, gpu_index: int, assigned_uuid: str) -> bool:
    rows = [row.strip().split(",", 4) for row in snapshot.splitlines() if row.strip()]
    matches = [row for row in rows if len(row) == 5 and int(row[0].strip()) == gpu_index]
    if len(matches) != 1:
        raise ValueError("assigned GPU index not found")
    row = matches[0]
    if row[1].strip() != assigned_uuid:
        raise ValueError("GPU UUID at assigned index changed")
    used_mib = int(row[3].strip().split()[0])
    utilization = int(row[4].strip().split()[0])
    return used_mib < 1000 and utilization < 10


def validate_training_receipt(receipt: dict, gpu_name_fragment: str) -> None:
    if receipt.get("status") != "DONE":
        raise ValueError("training not DONE")
    if receipt.get("optimizer_steps") != receipt.get("optimizer_steps_expected") or not receipt.get("optimizer_steps"):
        raise ValueError("optimizer steps incomplete")
    if not math.isfinite(float(receipt.get("mean_loss", float("nan")))):
        raise ValueError("non-finite training loss")
    if receipt.get("runtime_compute_dtype") != "torch.float16" or receipt.get("runtime_master_dtype") != "torch.float32":
        raise ValueError("training precision differs from amended protocol")
    if gpu_name_fragment.replace("_", " ") not in receipt.get("cuda_device_name", ""):
        raise ValueError("training GPU type differs from assignment")


def validate_evaluation_receipt(status: dict, summary: dict) -> None:
    if status.get("status") != "DONE" or status.get("tasks") != 1000:
        raise ValueError("ranking status incomplete")
    if summary.get("n_tasks") != 1000 or summary.get("candidate_program_scores") != 125000:
        raise ValueError("ranking program count incomplete")
    if not math.isfinite(float(summary.get("hit@32", float("nan")))):
        raise ValueError("non-finite Hit@32")


def job_name(k: int, subset: int, mode: str, seed: int, arm: str) -> str:
    return f"verifier_withheld_{'train' if mode == 'train' else 'eval'}_20260923_k{k}s{subset}_{seed}_{arm}"


def docker_inspect(name: str) -> dict | None:
    result = command(["docker", "inspect", name], allow_failure=True)
    if result.returncode:
        if "No such object" in result.stderr:
            return None
        raise RuntimeError(f"docker inspect failed: {result.stderr.strip()}")
    return json.loads(result.stdout)[0]


def assert_container_binding(inspect: dict, gpu_index: int) -> None:
    requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
    ids = [str(device) for request in requests for device in request.get("DeviceIDs", [])]
    if ids != [str(gpu_index)]:
        raise ValueError(f"container GPU binding differs: {ids}")


def capture_accepted_if_exited(name: str, inspect: dict | None, output: Path) -> None:
    if inspect is not None and inspect["State"]["Status"] == "exited":
        capture_container(name, output)


def validate_runner_assignment(runner: str, gpu_name: str) -> None:
    normalized_name = gpu_name.replace("_", " ")
    if runner not in ("run_first.py", "run_amended.py"):
        raise ValueError("unknown runner")
    if runner == "run_first.py" and "V100" not in normalized_name:
        raise ValueError("original runner requires V100")
    if runner == "run_amended.py" and "RTX 3060" not in normalized_name:
        raise ValueError("amended runner requires RTX 3060")


def job_accepted(output: Path, k: int, subset: int, mode: str, seed: int,
                 arm: str, gpu_name_fragment: str) -> bool:
    prefix = f"k{k}_s{subset}_seed{seed}_{arm}"
    if mode == "train":
        receipt_path = output / "adapters" / prefix / "training_receipt.json"
        if not receipt_path.exists():
            if (output / "adapters" / (prefix + ".partial")).exists():
                raise ValueError("partial training adapter requires audit")
            return False
        validate_training_receipt(json.loads(receipt_path.read_text()), gpu_name_fragment)
        status_path = output / "runs/training" / f"adapters__{prefix}.status.json"
        if not status_path.exists() or json.loads(status_path.read_text()).get("status") != "DONE":
            raise ValueError("training status missing or incomplete")
        if arm == "withheld":
            pair_path = output / "runs" / f"k{k}_s{subset}_seed{seed}_pair.json"
            if not pair_path.exists():
                raise ValueError("paired training acceptance missing")
            pair = json.loads(pair_path.read_text())
            if pair.get("status") != "DONE" or pair.get("freeze_sha256") != FREEZE_SHA256:
                raise ValueError("paired training acceptance invalid")
        return True
    statuses = []
    for split in ("withheld", "control"):
        branch = f"{prefix}_{split}"
        status_path = output / "rankings/status" / f"{branch}.json"
        summary_path = output / "rankings/summaries" / f"{branch}.json"
        if not status_path.exists() or not summary_path.exists():
            if statuses or status_path.exists() or summary_path.exists():
                raise ValueError("partial evaluation requires audit")
            return False
        status = json.loads(status_path.read_text())
        summary = json.loads(summary_path.read_text())
        validate_evaluation_receipt(status, summary)
        statuses.append(status)
    atomic_path = output / "rankings/atomic" / f"{prefix}_atomic.json"
    if not atomic_path.exists():
        raise ValueError("atomic evaluation missing")
    atomic = json.loads(atomic_path.read_text())
    if atomic.get("status") != "DONE" or atomic.get("tasks", 0) <= 0:
        raise ValueError("atomic evaluation incomplete")
    return True


def start_job(host_base: Path, gpu_index: int, k: int, subset: int, mode: str,
              seed: int, arm: str, uid_gid: str, pythonpath: str,
              runner: str, assigned_uuid: str) -> str:
    name = job_name(k, subset, mode, seed, arm)
    args = ["docker", "run", "-d", "--name", name, "--gpus", f"device={gpu_index}",
            "--user", uid_gid, "-v", f"{host_base}:/work",
            "-e", f"PYTHONPATH={pythonpath}", "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TOKENIZERS_PARALLELISM=false",
            "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "-e", f"ASSIGNED_GPU_UUID={assigned_uuid}", IMAGE, "python", "-u",
            f"{CONTAINER_REPO}/reports/withheld_pairs_20260923/{runner}",
            "--mode", mode, "--k", str(k), "--subset", str(subset),
            "--seed", str(seed), "--arm", arm,
            "--export", "/work/remote_resume_20260922/export"]
    return command(args).stdout.strip()


def run_queue(args: argparse.Namespace) -> None:
    import fcntl

    output = args.host_base / "withheld_pairs_20260923/repo/artifacts/withheld_pairs_20260923"
    if sha256(output / "FREEZE.json") != FREEZE_SHA256:
        raise RuntimeError("original frozen manifest changed")
    validate_runner_assignment(args.runner, args.gpu_name)
    if sha256(args.host_base / "withheld_pairs_20260923/repo/reports/withheld_pairs_20260923" / args.runner) != args.runner_sha256:
        raise RuntimeError("runner SHA-256 mismatch")
    allocation = {
        "schema": "withheld-pairs.gpu-allocation.v1", "k": args.k, "subset": args.subset,
        "host": args.host, "gpu_index": args.gpu_index, "gpu_uuid": args.gpu_uuid,
        "gpu_name": args.gpu_name, "compute_dtype": "torch.float16",
        "master_dtype": "torch.float32", "runner": args.runner,
        "runner_sha256": args.runner_sha256, "freeze_sha256": FREEZE_SHA256,
    }
    write_once(output / "runs" / f"k{args.k}_s{args.subset}_allocation.json",
               (json.dumps(allocation, sort_keys=True) + "\n").encode())
    event_path = output / "runs" / f"queue_k{args.k}s{args.subset}.jsonl"
    lock_path = output / "queue_next.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for mode, seed, arm in ordered_jobs():
            name = job_name(args.k, args.subset, mode, seed, arm)
            existing = docker_inspect(name)
            if existing is not None:
                assert_container_binding(existing, args.gpu_index)
            accepted = job_accepted(output, args.k, args.subset, mode, seed, arm, args.gpu_name)
            state = existing["State"] if existing else None
            actual_uuid = args.gpu_uuid if existing else None
            decision = resume_decision(accepted, state, actual_uuid, args.gpu_uuid)
            if decision == "SKIP":
                capture_accepted_if_exited(name, existing, output)
                log_event(event_path, {"status": "SKIPPED_ACCEPTED", "job": name})
                continue
            if decision == "VERIFY":
                capture_container(name, output)
                raise RuntimeError(f"exited job lacks valid acceptance: {name}")
            if decision == "START":
                if dt.datetime.now(dt.timezone.utc) >= args.deadline:
                    log_event(event_path, {"status": "STOPPED_AT_DEADLINE", "next_job": name})
                    return
                snapshot = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                                    "--format=csv,noheader"]).stdout
                if not gpu_is_available(snapshot, args.gpu_index, args.gpu_uuid):
                    raise RuntimeError("assigned GPU is occupied; refusing to launch")
                container_id = start_job(args.host_base, args.gpu_index, args.k, args.subset,
                                         mode, seed, arm, args.uid_gid, args.pythonpath,
                                         args.runner, args.gpu_uuid)
                log_event(event_path, {"status": "STARTED", "job": name, "container_id": container_id})
            while True:
                current = docker_inspect(name)
                if current is None:
                    raise RuntimeError(f"container disappeared: {name}")
                assert_container_binding(current, args.gpu_index)
                state = current["State"]
                if state["Status"] == "exited":
                    capture_container(name, output)
                    if state["ExitCode"] != 0:
                        log_event(event_path, {"status": "FAILED", "job": name,
                                               "exit_code": state["ExitCode"]})
                        raise RuntimeError(f"job failed: {name}")
                    if not job_accepted(output, args.k, args.subset, mode, seed, arm, args.gpu_name):
                        raise RuntimeError(f"job exited zero without acceptance: {name}")
                    log_event(event_path, {"status": "DONE", "job": name})
                    break
                if state["Status"] != "running":
                    raise RuntimeError(f"unexpected container state: {state['Status']}")
                time.sleep(20)
        log_event(event_path, {"status": "SET_DONE", "k": args.k, "subset": args.subset,
                               "jobs": len(ordered_jobs())})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-base", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--uid-gid", required=True)
    parser.add_argument("--pythonpath", required=True)
    parser.add_argument("--runner", choices=("run_first.py", "run_amended.py"), required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--subset", type=int, required=True)
    parser.add_argument("--deadline", type=dt.datetime.fromisoformat, required=True)
    args = parser.parse_args()
    if args.deadline.tzinfo is None:
        parser.error("deadline must include a time zone")
    run_queue(args)


if __name__ == "__main__":
    main()

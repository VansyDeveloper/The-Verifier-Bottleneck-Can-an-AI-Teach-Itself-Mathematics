"""Host-side, single-GPU queue for the preregistered first pair set.

It attaches to existing containers, never removes them, and stops on the first
failed or unverified job. This file does not change the frozen model protocol.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path


JOBS = [(mode, seed, arm) for mode in ("train", "evaluate")
        for seed in (85000, 85001, 85002) for arm in ("random", "withheld")]
FREEZE_SHA256 = "99DF70CE2030C18F3E86368E1C4A86EA3704FA729C6C355EE8A90C90D1333DD2"
IMAGE = "dcs-spatial-v3:torch2.7.1-cu118"
HOST_BASE = Path("/home/rovnyago_dv/verifier_bottleneck_20260922")
CONTAINER_BASE = "/work/withheld_pairs_20260923/repo"
DEADLINE = dt.datetime(2026, 9, 23, 4, 33, tzinfo=dt.timezone.utc)


def command(args: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode and not allow_failure:
        raise RuntimeError(f"command failed: {args[0]} {args[1:3]}: {result.stderr.strip()}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def gpu0_available(output: str) -> bool:
    rows = [line.strip().split(",", 1) for line in output.splitlines() if line.strip()]
    if not rows or rows[0][0].strip() != "0":
        raise ValueError("nvidia-smi GPU 0 identity missing")
    used = int(rows[0][1].strip().split()[0])
    return used < 1000


def validate_training_receipt(receipt: dict) -> None:
    if receipt.get("status") != "DONE":
        raise ValueError("training receipt is not DONE")
    if receipt.get("optimizer_steps") != receipt.get("optimizer_steps_expected") or receipt["optimizer_steps"] <= 0:
        raise ValueError("training optimizer steps incomplete")
    if receipt.get("runtime_compute_dtype") != "torch.float16":
        raise ValueError("training did not use V100 FP16 protocol")
    if not math.isfinite(float(receipt["mean_loss"])):
        raise ValueError("non-finite training mean loss")


def job_name(mode: str, seed: int, arm: str) -> str:
    return f"verifier_withheld_{'train' if mode == 'train' else 'eval'}_20260923_k1s1_{seed}_{arm}"


def state(name: str) -> dict | None:
    result = command(["docker", "inspect", name], allow_failure=True)
    if result.returncode:
        if "No such object" in result.stderr:
            return None
        raise RuntimeError(f"docker inspect failed for {name}: {result.stderr.strip()}")
    return json.loads(result.stdout)[0]["State"]


def log_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **event},
                                sort_keys=True) + "\n")
    print(json.dumps(event, sort_keys=True), flush=True)


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing job evidence differs: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(payload)


def capture_container(name: str, output: Path) -> None:
    base = output / "runs/docker" / name
    inspect = command(["docker", "inspect", name]).stdout.encode()
    logs = command(["docker", "logs", name])
    write_once(base.with_suffix(".inspect.json"), inspect)
    write_once(base.with_suffix(".stdout.log"), logs.stdout.encode())
    write_once(base.with_suffix(".stderr.log"), logs.stderr.encode())


def verify_job(mode: str, seed: int, arm: str, output: Path) -> None:
    prefix = f"k1_s1_seed{seed}_{arm}"
    if mode == "train":
        receipt = json.loads((output / "adapters" / prefix / "training_receipt.json").read_text())
        validate_training_receipt(receipt)
        status = json.loads((output / "runs/training" / f"adapters__{prefix}.status.json").read_text())
        if status.get("status") != "DONE":
            raise ValueError(f"training status not DONE: {prefix}")
        if arm == "withheld":
            pair = json.loads((output / "runs" / f"k1_s1_seed{seed}_pair.json").read_text())
            if pair.get("status") != "DONE" or pair.get("freeze_sha256") != FREEZE_SHA256:
                raise ValueError(f"paired training acceptance failed: {seed}")
    else:
        for split in ("withheld", "control"):
            branch = f"{prefix}_{split}"
            status = json.loads((output / "rankings/status" / f"{branch}.json").read_text())
            summary = json.loads((output / "rankings/summaries" / f"{branch}.json").read_text())
            if status.get("status") != "DONE" or status.get("tasks") != 1000:
                raise ValueError(f"ranking incomplete: {branch}")
            if summary.get("n_tasks") != 1000 or summary.get("candidate_program_scores") != 125000:
                raise ValueError(f"ranking candidate count incomplete: {branch}")
            if not math.isfinite(float(summary["hit@32"])):
                raise ValueError(f"non-finite Hit@32: {branch}")
        atomic = json.loads((output / "rankings/atomic" / f"{prefix}_atomic.json").read_text())
        if atomic.get("status") != "DONE" or atomic.get("tasks", 0) <= 0:
            raise ValueError(f"atomic evaluation incomplete: {prefix}")


def start_job(mode: str, seed: int, arm: str, host_base: Path) -> str:
    output = command(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"])
    if not gpu0_available(output.stdout):
        raise RuntimeError("V100 GPU 0 is occupied; refusing to launch")
    name = job_name(mode, seed, arm)
    args = ["docker", "run", "-d", "--name", name, "--gpus", "device=0",
            "--user", "1045:1046", "-v", f"{host_base}:/work",
            "-e", "PYTHONPATH=/work/python_pkgs_eval", "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TOKENIZERS_PARALLELISM=false",
            "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8", IMAGE, "python", "-u",
            f"{CONTAINER_BASE}/reports/withheld_pairs_20260923/run_first.py",
            "--mode", mode, "--k", "1", "--subset", "1", "--seed", str(seed),
            "--arm", arm, "--export", "/work/remote_resume_20260922/export"]
    return command(args).stdout.strip()


def run_queue(host_base: Path, deadline: dt.datetime) -> None:
    import fcntl
    output = host_base / "withheld_pairs_20260923/repo/artifacts/withheld_pairs_20260923"
    if sha256(output / "FREEZE.json") != FREEZE_SHA256:
        raise RuntimeError("frozen manifest changed")
    lock_path = output / "queue_first.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        event_path = output / "runs/queue_first.jsonl"
        for mode, seed, arm in JOBS:
            name = job_name(mode, seed, arm)
            present = state(name)
            if present is None:
                if dt.datetime.now(dt.timezone.utc) >= deadline:
                    log_event(event_path, {"status": "STOPPED_AT_DEADLINE", "next_job": name})
                    return
                container_id = start_job(mode, seed, arm, host_base)
                log_event(event_path, {"status": "STARTED", "job": name, "container_id": container_id})
            while True:
                current = state(name)
                if current is None:
                    raise RuntimeError(f"container disappeared during queue: {name}")
                if current["Status"] == "exited":
                    if current["ExitCode"] != 0:
                        log_event(event_path, {"status": "FAILED", "job": name,
                                               "exit_code": current["ExitCode"]})
                        capture_container(name, output)
                        raise RuntimeError(f"job failed: {name}")
                    capture_container(name, output)
                    verify_job(mode, seed, arm, output)
                    log_event(event_path, {"status": "DONE", "job": name})
                    break
                if current["Status"] != "running":
                    raise RuntimeError(f"unexpected container state {current['Status']}: {name}")
                time.sleep(20)
        log_event(event_path, {"status": "FIRST_BLOCK_JOBS_DONE", "jobs": len(JOBS)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-base", type=Path, default=HOST_BASE)
    parser.add_argument("--deadline", default=DEADLINE.isoformat())
    args = parser.parse_args()
    run_queue(args.host_base, dt.datetime.fromisoformat(args.deadline))


if __name__ == "__main__":
    main()

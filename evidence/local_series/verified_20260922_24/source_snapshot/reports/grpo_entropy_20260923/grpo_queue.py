"""Host-side resumable queue for complete same-GPU entropy-ablation pairs.

Accepted Docker jobs are never restarted. Failed or partial jobs and their
containers remain intact for audit. This coordinator requires only Python's
standard library on the university host.
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


IMAGE = "dcs-spatial-v3:torch2.7.1-cu118"
TRAIN_CODE_SHA = "5E0D5C1FE53BE54E75F0BEA01BF94A865DD3F73827EF47409482B8EE14299188"
EVAL_CODE_SHA = "815B78726B9C0B8053FE87CB9D016A979D6F98A8B94C5DAC4717D385238374A5"
BASE_SHA = "CD2A512003E2F9F3CD3C32A9C3573F820BB28C940F73C57B1DDAA983D9223EBA"
ADAPTER_SHA = "C47403B7F6474C4C1261572DA0253CDBCE576AE22AADCFE86DD64B20254B3E04"
INPUT_SHA = "140E95B8192D0E19E79716385FCDD2361483B07990324F7F933C007AACE7715F"
HOLDOUT_SHA = "703049FB3B5655AFED95B2BADA36B3C85ADC2712EF5C03A8DBF54A12AFA9C57D"
DURATION_SHA = "2D3BA9B90F376CD81A49605F8ABB9175F649C765494B6F5C5B512F84D81C2B91"
REL = Path("grpo_entropy_20260923/repo")
CONTAINER_ROOT = "/work/grpo_entropy_20260923/repo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((item for item in root.rglob("*") if item.is_file()),
                   key=lambda item: item.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"empty adapter directory: {root}")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest().upper()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing evidence differs: {path}")
        return
    path.write_bytes(data)


def event(path: Path, status: str, **details) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": status, **details}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def ordered_jobs(seeds: tuple[int, ...]) -> list[tuple[int, str, str]]:
    return [(seed, mode, arm) for seed in seeds for mode in ("train", "evaluate")
            for arm in ("control", "entropy")]


def resume_decision(accepted: bool, state: dict | None) -> str:
    if state is None:
        if accepted:
            raise ValueError("accepted run lacks its Docker evidence")
        return "START"
    if state.get("Status") == "running":
        return "WAIT"
    if state.get("Status") == "exited":
        if state.get("ExitCode") != 0:
            raise ValueError("prior container failed; preserve and audit it")
        if not accepted:
            raise ValueError("exited-zero container lacks acceptance")
        return "SKIP"
    raise ValueError(f"unexpected container state: {state.get('Status')}")


def validate_pair(control: dict, entropy: dict) -> None:
    if control.get("entropy_coef") != 0.0 or entropy.get("entropy_coef") != 0.01:
        raise ValueError("paired entropy coefficients differ from protocol")
    left = {key: value for key, value in control.items() if key != "entropy_coef"}
    right = {key: value for key, value in entropy.items() if key != "entropy_coef"}
    if left != right:
        raise ValueError("unequal paired training configuration")


def container_name(seed: int, mode: str, arm: str) -> str:
    return f"verifier_grpo_entropy_{run_name(seed, arm)}{'_eval' if mode == 'evaluate' else ''}"


def run_name(seed: int, arm: str) -> str:
    return f"seed{seed}_{arm}_fp32"


def queue_paths(output_root: Path, gpu_index: int) -> tuple[Path, Path, Path]:
    prefix = output_root / f"queue_gpu{gpu_index}_fp32"
    return (prefix.with_name(prefix.name + "_allocation.json"),
            prefix.with_suffix(".jsonl"), prefix.with_suffix(".lock"))


def require_frozen_steps(freeze: dict, steps: int) -> None:
    if (freeze.get("status") != "FROZEN_BEFORE_FULL_PAIR" or
            freeze.get("steps_per_arm") != steps or
            freeze.get("heldout_outcomes_viewed_before_decision") is not False):
        raise ValueError("requested steps differ from blind duration freeze")


def docker_command(host_base: Path, gpu_index: int, gpu_uuid: str, seed: int,
                   mode: str, arm: str, uid_gid: str, steps: int) -> list[str]:
    repo = CONTAINER_ROOT
    model = "/work/grpo_entropy_20260923/models/Qwen3-0.6B-Base"
    common = ["docker", "run", "-d", "--name", container_name(seed, mode, arm),
              "--gpus", f"device={gpu_index}", "--user", uid_gid,
              "-v", f"{host_base}:/work",
              "-e", f"PYTHONPATH=/work/python_pkgs_eval:{repo}/src",
              "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1",
              "-e", "TOKENIZERS_PARALLELISM=false", "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
              IMAGE, "python", "-u"]
    name = run_name(seed, arm)
    output = f"{repo}/artifacts/grpo_entropy_20260923/runs"
    if mode == "train":
        return common + [f"{repo}/reports/grpo_entropy_20260923/run_ablation.py",
                         "--run-name", name, "--output-root", output,
                         "--base-model", model,
                         "--adapter", f"{repo}/artifacts/adapters/sft_atomic_r32_pilot_aw4_cont",
                         "--input", f"{repo}/artifacts/data/pilot/rl_train.jsonl",
                         "--expected-base-sha256", BASE_SHA,
                         "--expected-adapter-sha256", ADAPTER_SHA,
                         "--expected-input-sha256", INPUT_SHA,
                         "--gpu-uuid", gpu_uuid, "--seed", str(seed),
                         "--steps", str(steps), "--entropy-coef", "0" if arm == "control" else "0.01"]
    if mode == "evaluate":
        return common + [f"{repo}/reports/grpo_entropy_20260923/run_evaluation.py",
                         "--run-name", name, "--output-root", output,
                         "--base-model", model,
                         "--holdout", f"{repo}/artifacts/grpo_entropy_20260923/holdout_1000.jsonl",
                         "--expected-holdout-sha256", HOLDOUT_SHA,
                         "--gpu-uuid", gpu_uuid, "--shard-size", "25"]
    raise ValueError(f"unknown mode: {mode}")


def command(args: list[str], allow_failure: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode and not allow_failure:
        raise RuntimeError(f"command failed {args[:3]}: {result.stderr.strip()}")
    return result


def inspect_container(name: str) -> dict | None:
    result = command(["docker", "inspect", name], allow_failure=True)
    if result.returncode:
        if "No such object" in result.stderr:
            return None
        raise RuntimeError(f"docker inspect failed: {result.stderr.strip()}")
    return json.loads(result.stdout)[0]


def assert_binding(info: dict, gpu_index: int) -> None:
    requests = info.get("HostConfig", {}).get("DeviceRequests") or []
    ids = [str(device) for request in requests for device in request.get("DeviceIDs", [])]
    if ids != [str(gpu_index)]:
        raise ValueError(f"container GPU binding differs: {ids}")


def gpu_free(gpu_index: int, gpu_uuid: str) -> bool:
    snapshot = command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                        "--format=csv,noheader"]).stdout
    rows = [line.split(",", 3) for line in snapshot.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 4 and int(row[0].strip()) == gpu_index]
    if len(matches) != 1 or matches[0][1].strip() != gpu_uuid:
        raise ValueError("assigned GPU UUID or index changed")
    memory = int(matches[0][2].strip().split()[0])
    utilization = int(matches[0][3].strip().split()[0])
    return memory < 1000 and utilization < 10


def capture(name: str, output_root: Path, info: dict) -> None:
    path = output_root / "queue_evidence"
    path.mkdir(parents=True, exist_ok=True)
    logs = command(["docker", "logs", name])
    write_once(path / f"{name}.stdout.log", logs.stdout.encode())
    write_once(path / f"{name}.stderr.log", logs.stderr.encode())
    write_once(path / f"{name}.inspect.json",
               (json.dumps(info, sort_keys=True, indent=2) + "\n").encode())


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def accepted(output_root: Path, seed: int, mode: str, arm: str, gpu_uuid: str,
             steps: int) -> bool:
    run = output_root / "runs" / run_name(seed, arm)
    if (run / "FAILED.json").exists() or (run / "evaluation/FAILED.json").exists():
        raise ValueError("failed run requires audit")
    training_done = run / "DONE.json"
    if not training_done.exists():
        return False
    receipt = json.loads(training_done.read_text(encoding="utf-8"))
    config = receipt["config"]
    if (receipt.get("status") != "DONE" or receipt.get("steps") != steps or
            receipt.get("groups") != steps or receipt.get("answers") != 8 * steps or
            config.get("seed") != seed or config.get("gpu_uuid") != gpu_uuid or
            config.get("entropy_coef") != (0.0 if arm == "control" else 0.01) or
            config.get("gradient_checkpointing") != "non_reentrant_trainable_model" or
            config.get("input_sha256") != INPUT_SHA or config.get("base_sha256") != BASE_SHA or
            config.get("adapter_sha256") != ADAPTER_SHA):
        raise ValueError("training receipt differs from frozen protocol")
    metrics = run / "metrics.jsonl"
    generations = run / "generations.jsonl"
    if (sha256(metrics) != receipt["metrics_sha256"] or _line_count(metrics) != steps or
            sha256(generations) != receipt["generations_sha256"] or _line_count(generations) != 8 * steps or
            tree_sha256(run / "final_adapter") != receipt["adapter_tree_sha256"]):
        raise ValueError("training raw artifacts or checkpoint integrity failed")
    if mode == "train":
        return True
    evaluation_done = run / "evaluation/DONE.json"
    if not evaluation_done.exists():
        return False
    summary = json.loads(evaluation_done.read_text(encoding="utf-8"))
    hit_k = summary.get("hit@k")
    if (summary.get("status") != "DONE" or summary.get("tasks") != 1000 or
            summary.get("candidate_program_scores") != 125000 or
            summary.get("holdout_sha256") != HOLDOUT_SHA or
            summary.get("adapter_tree_sha256") != receipt["adapter_tree_sha256"] or
            summary.get("gpu_uuid") != gpu_uuid or
            not isinstance(hit_k, dict) or len(hit_k) != 125 or
            not math.isfinite(float(summary.get("hit@32", float("nan")))) or
            summary["hit@32"] != hit_k.get("32")):
        raise ValueError("evaluation receipt differs from frozen protocol")
    task_ids = []
    for index in range(40):
        base = run / "evaluation/shards" / f"part-{index:05d}"
        shard = base.with_suffix(".jsonl.gz")
        shard_receipt = base.with_suffix(".receipt.json")
        record = json.loads(shard_receipt.read_text(encoding="utf-8"))
        if (record.get("status") != "DONE" or record.get("tasks") != 25 or
                len(record.get("task_ids", [])) != 25 or sha256(shard) != record.get("sha256")):
            raise ValueError("evaluation shard SHA-256 or task count differs")
        task_ids.extend(record["task_ids"])
    if len(set(task_ids)) != 1000:
        raise ValueError("evaluation has duplicated tasks")
    return True


def verify_sources(host_base: Path) -> None:
    repo = host_base / REL
    paths = {
        repo / "reports/grpo_entropy_20260923/run_ablation.py": TRAIN_CODE_SHA,
        repo / "reports/grpo_entropy_20260923/run_evaluation.py": EVAL_CODE_SHA,
        repo / "artifacts/data/pilot/rl_train.jsonl": INPUT_SHA,
        repo / "artifacts/grpo_entropy_20260923/holdout_1000.jsonl": HOLDOUT_SHA,
        repo / "reports/grpo_entropy_20260923/DURATION_FREEZE_FP32.json": DURATION_SHA,
        host_base / "grpo_entropy_20260923/models/Qwen3-0.6B-Base/model.safetensors": BASE_SHA,
    }
    for path, expected in paths.items():
        if sha256(path) != expected:
            raise RuntimeError(f"source SHA-256 mismatch: {path}")
    if tree_sha256(repo / "artifacts/adapters/sft_atomic_r32_pilot_aw4_cont") != ADAPTER_SHA:
        raise RuntimeError("initial adapter tree SHA-256 mismatch")


def launch_path_is_clear(output_root: Path, seed: int, mode: str, arm: str) -> None:
    run = output_root / "runs" / run_name(seed, arm)
    if mode == "train":
        if run.exists():
            raise ValueError(f"partial training run without Docker container: {run}")
        return
    if mode == "evaluate":
        if not (run / "DONE.json").exists():
            raise ValueError("evaluation requires accepted training")
        if (run / "evaluation").exists():
            raise ValueError(f"partial evaluation without Docker container: {run}")
        return
    raise ValueError(f"unknown job mode: {mode}")


def run_queue(args: argparse.Namespace) -> None:
    import fcntl

    output_root = args.host_base / REL / "artifacts/grpo_entropy_20260923"
    verify_sources(args.host_base)
    freeze = json.loads((args.host_base / REL / "reports/grpo_entropy_20260923/DURATION_FREEZE_FP32.json")
                        .read_text(encoding="utf-8"))
    require_frozen_steps(freeze, args.steps)
    allocation = {"schema": "grpo-entropy.gpu-queue.v1", "seeds": list(args.seeds),
                  "gpu_index": args.gpu_index, "gpu_uuid": args.gpu_uuid,
                  "steps_per_arm": args.steps, "base_sha256": BASE_SHA,
                  "adapter_sha256": ADAPTER_SHA, "input_sha256": INPUT_SHA,
                  "holdout_sha256": HOLDOUT_SHA, "train_code_sha256": TRAIN_CODE_SHA,
                  "eval_code_sha256": EVAL_CODE_SHA,
                  "duration_freeze_sha256": DURATION_SHA}
    allocation_path, events, lock_path = queue_paths(output_root, args.gpu_index)
    write_once(allocation_path,
               (json.dumps(allocation, sort_keys=True) + "\n").encode())
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed, mode, arm in ordered_jobs(args.seeds):
            name = container_name(seed, mode, arm)
            info = inspect_container(name)
            if info is not None:
                assert_binding(info, args.gpu_index)
            is_accepted = accepted(output_root, seed, mode, arm, args.gpu_uuid, args.steps)
            decision = resume_decision(is_accepted, info["State"] if info else None)
            if decision == "SKIP":
                capture(name, output_root, info)
                event(events, "SKIPPED_ACCEPTED", job=name)
                continue
            if decision == "START":
                launch_path_is_clear(output_root, seed, mode, arm)
                if dt.datetime.now(dt.timezone.utc) >= args.latest_start:
                    event(events, "STOPPED_AT_DEADLINE", next_job=name)
                    return
                for _ in range(12):
                    if gpu_free(args.gpu_index, args.gpu_uuid):
                        break
                    time.sleep(5)
                else:
                    raise RuntimeError("assigned GPU is occupied; refusing new launch")
                started = command(docker_command(args.host_base, args.gpu_index,
                                                 args.gpu_uuid, seed, mode, arm,
                                                 args.uid_gid, args.steps)).stdout.strip()
                event(events, "STARTED", job=name, container_id=started)
            while True:
                current = inspect_container(name)
                if current is None:
                    raise RuntimeError(f"active container disappeared: {name}")
                assert_binding(current, args.gpu_index)
                state = current["State"]
                if state["Status"] == "exited":
                    capture(name, output_root, current)
                    if state["ExitCode"] != 0:
                        event(events, "FAILED", job=name, exit_code=state["ExitCode"])
                        raise RuntimeError(f"job failed: {name}")
                    if not accepted(output_root, seed, mode, arm, args.gpu_uuid, args.steps):
                        raise RuntimeError(f"exited-zero job lacks acceptance: {name}")
                    event(events, "DONE", job=name)
                    break
                if state["Status"] != "running":
                    raise RuntimeError(f"unexpected container state: {state['Status']}")
                time.sleep(20)
            if mode == "train" and arm == "entropy":
                control = json.loads((output_root / "runs" / run_name(seed, "control") / "config.json").read_text())
                treatment = json.loads((output_root / "runs" / run_name(seed, "entropy") / "config.json").read_text())
                validate_pair(control, treatment)
                event(events, "PAIR_TRAINING_ACCEPTED", seed=seed)
        event(events, "QUEUE_DONE", seeds=list(args.seeds), jobs=len(ordered_jobs(args.seeds)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-base", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--uid-gid", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, choices=(150, 400), required=True)
    parser.add_argument("--latest-start", type=dt.datetime.fromisoformat, required=True)
    args = parser.parse_args()
    if args.latest_start.tzinfo is None or args.gpu_index not in (1, 2):
        parser.error("assigned university GPU and timezone-aware deadline required")
    run_queue(args)


if __name__ == "__main__":
    main()

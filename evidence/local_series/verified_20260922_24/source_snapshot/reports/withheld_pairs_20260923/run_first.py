"""Run one frozen withheld-pair training or evaluation job on approved V100."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import traceback
import types
from pathlib import Path

from fp16_training import fp16_runtime, train_branch_fp16
from prepare import json_bytes, read_jsonl, sha256, write_exclusive_or_verify


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8"
ATOMIC = REPO / "artifacts/stage4_sh1_v5_0p6b"
CONFIRM = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
OUTPUT = REPO / "artifacts/withheld_pairs_20260923"
SOURCE = LEGACY / "data/discover_trajectories.jsonl"
sys.path.insert(0, str(LEGACY / "code"))
sys.modules.setdefault("v5_train", types.ModuleType("v5_train"))


def validate_pair_receipts(left: dict, right: dict) -> dict:
    if left.get("status") != "DONE" or right.get("status") != "DONE":
        raise ValueError("both training receipts must be DONE")
    if not left.get("atomic_export_sha256") or not right.get("atomic_export_sha256"):
        raise ValueError("paired atomic export binding missing")
    keys = ("optimizer_steps", "effective_batch", "epochs", "records",
            "loss_bearing_target_tokens_seen", "runtime_compute_dtype",
            "atomic_export_sha256")
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError("paired training budget or runtime differs")
    if left.get("runtime_compute_dtype") != "torch.float16":
        raise ValueError("paired training is not FP16")
    if not math.isfinite(float(left["mean_loss"])) or not math.isfinite(float(right["mean_loss"])):
        raise ValueError("non-finite paired training loss")
    return {key: left.get(key) for key in keys}


def entry_for(freeze: dict, k: int, subset: int) -> dict:
    entries = [row for row in freeze["entries"] if row["k"] == k and row["subset"] == subset]
    if len(entries) != 1:
        raise RuntimeError("requested pair set absent from freeze")
    entry = entries[0]
    if entry["status"] != "FEASIBLE":
        raise RuntimeError(f"preselected pair set is infeasible: {entry.get('reason')}")
    return entry


def verify_freeze() -> dict:
    freeze_path = OUTPUT / "FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or freeze.get("schema") != "verifier-bottleneck.withheld-pairs-freeze.v1":
        raise RuntimeError("withheld-pairs freeze invalid")
    if sha256(SOURCE) != freeze["source_sha256"]:
        raise RuntimeError("source trajectory SHA-256 changed")
    for relative, expected in freeze["code_sha256"].items():
        if sha256(REPO / relative) != expected:
            raise RuntimeError(f"frozen code changed: {relative}")
    for entry in freeze["entries"]:
        if entry["status"] == "FEASIBLE":
            for holdout in entry["holdouts"].values():
                if sha256(OUTPUT / holdout["path"]) != holdout["sha256"]:
                    raise RuntimeError("frozen holdout SHA-256 changed")
    return freeze


@contextlib.contextmanager
def run_lease():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "orchestrator.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another withheld-pairs job holds the lease") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another withheld-pairs job holds the lease") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_libraries(freeze: dict):
    import composition_core as core
    import composition_eval as eval_lib
    import composition_model as model_lib

    fp16_runtime(freeze["training_config"])
    model_lib.ROOT = OUTPUT
    model_lib.PROTOCOL = {"model": "Qwen/Qwen3-0.6B", "pilot_initial": freeze["training_config"]}
    return core, model_lib, eval_lib


def balanced_atomic_pool(core, seed: int):
    import random

    sh1 = read_jsonl(ATOMIC / "data/full_sh1.jsonl")
    controls = read_jsonl(ATOMIC / "data/control_apply.jsonl")
    by_operation = {"SH1": sh1}
    for operation in core.OPS[1:]:
        by_operation[operation] = [row for row in controls if row["operation"] == operation]
    rng = random.Random(seed + 1000)
    rows = []
    for operation in core.OPS:
        rows.extend(rng.sample(by_operation[operation], 1000))
    rng.shuffle(rows)
    return rows


def prepared_pair(freeze: dict, entry: dict, seed: int, core, model_lib, export: Path) -> tuple[dict, dict]:
    source = {row["trajectory_id"]: row for row in read_jsonl(SOURCE)}
    selection = entry["selections"][str(seed)]
    tokenizer, _ = model_lib.tokenizer_and_ids(export)
    atomic_pool = balanced_atomic_pool(core, seed)
    resolved = {**freeze["training_config"], "seed": seed,
                "schema": "withheld-pairs.v100-fp16-resolved.v1",
                "k": entry["k"], "subset": entry["subset"]}
    prepared = {}
    for arm in ("random", "withheld"):
        ids = selection[f"{arm}_ids"]
        trajectories = [source[identity] for identity in ids]
        specs = model_lib.distill_example_specs(trajectories, atomic_pool,
                                                 resolved["atomic_replay"], seed)
        encoded = model_lib.encode_specs(tokenizer, specs)
        prepared[arm] = {"encoded": encoded,
                         "hash": hashlib.sha256(json.dumps(encoded, ensure_ascii=True, sort_keys=True,
                                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest().upper(),
                         "target_tokens": sum(row["target_tokens"] for row in encoded)}
    if len(prepared["random"]["encoded"]) != len(prepared["withheld"]["encoded"]):
        raise RuntimeError("paired record counts differ before training")
    if prepared["random"]["target_tokens"] != prepared["withheld"]["target_tokens"]:
        raise RuntimeError("paired target-token budgets differ before training")
    if len(prepared["random"]["encoded"]) != 938:
        raise RuntimeError("training record count differs from frozen 750 plus 20% replay")
    return prepared, resolved


def adapter_path(k: int, subset: int, seed: int, arm: str) -> Path:
    return OUTPUT / "adapters" / f"k{k}_s{subset}_seed{seed}_{arm}"


def pair_receipt_path(k: int, subset: int, seed: int) -> Path:
    return OUTPUT / "runs" / f"k{k}_s{subset}_seed{seed}_pair.json"


def accept_pair(freeze: dict, entry: dict, seed: int, model_lib) -> dict | None:
    arms = {}
    for arm in ("random", "withheld"):
        adapter = adapter_path(entry["k"], entry["subset"], seed, arm)
        receipt_path = adapter / "training_receipt.json"
        if not receipt_path.exists():
            return None
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        arms[arm] = {"receipt": receipt, "receipt_sha256": sha256(receipt_path),
                     "tree_sha256": model_lib.tree_sha256(adapter)}
    budget = validate_pair_receipts(arms["random"]["receipt"], arms["withheld"]["receipt"])
    pair = {"schema": "withheld-pairs.training-pair.v1", "status": "DONE",
            "k": entry["k"], "subset": entry["subset"], "seed": seed,
            "freeze_sha256": sha256(OUTPUT / "FREEZE.json"), "budget": budget,
            "adapters": {arm: {key: arms[arm][key] for key in ("receipt_sha256", "tree_sha256")}
                         for arm in arms}}
    write_exclusive_or_verify(pair_receipt_path(entry["k"], entry["subset"], seed), json_bytes(pair))
    return pair


def train_one(freeze: dict, entry: dict, seed: int, arm: str, export: Path, core, model_lib) -> None:
    prepared, resolved = prepared_pair(freeze, entry, seed, core, model_lib, export)
    chosen = prepared[arm]
    output = adapter_path(entry["k"], entry["subset"], seed, arm)
    print(json.dumps({"event": "training_start", "seed": seed, "arm": arm,
                      "records": len(chosen["encoded"]), "target_tokens_per_epoch": chosen["target_tokens"]}), flush=True)
    receipt = train_branch_fp16(model_lib, export, output, chosen["encoded"], resolved, arm, chosen["hash"])
    print(json.dumps({"event": "training_done", "seed": seed, "arm": arm,
                      "optimizer_steps": receipt["optimizer_steps"], "mean_loss": receipt["mean_loss"]}), flush=True)
    pair = accept_pair(freeze, entry, seed, model_lib)
    if pair is not None:
        print(json.dumps({"event": "pair_accepted", "seed": seed, "budget": pair["budget"]}), flush=True)


def evaluate_one(freeze: dict, entry: dict, seed: int, arm: str, export: Path, model_lib, eval_lib) -> None:
    pair_path = pair_receipt_path(entry["k"], entry["subset"], seed)
    if not pair_path.is_file():
        raise RuntimeError("incomplete paired training cannot be evaluated")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    if pair.get("status") != "DONE" or pair["freeze_sha256"] != sha256(OUTPUT / "FREEZE.json"):
        raise RuntimeError("paired training receipt invalid")
    adapter = adapter_path(entry["k"], entry["subset"], seed, arm)
    if sha256(adapter / "training_receipt.json") != pair["adapters"][arm]["receipt_sha256"]:
        raise RuntimeError("adapter receipt hash mismatch")
    if model_lib.tree_sha256(adapter) != pair["adapters"][arm]["tree_sha256"]:
        raise RuntimeError("adapter tree hash mismatch")
    model, tokenizer, token_ids = model_lib.load_branch(export, adapter)
    try:
        binding = {"schema": "withheld-pairs.eval-binding.v1", "k": entry["k"],
                   "subset": entry["subset"], "seed": seed, "arm": arm,
                   "freeze_sha256": pair["freeze_sha256"],
                   "adapter_tree_sha256": pair["adapters"][arm]["tree_sha256"]}
        for kind in ("withheld", "control"):
            holdout = entry["holdouts"][kind]
            rows = read_jsonl(OUTPUT / holdout["path"])
            branch = f"k{entry['k']}_s{entry['subset']}_seed{seed}_{arm}_{kind}"
            summary, _ = eval_lib.evaluate_ranking(model, tokenizer, token_ids, rows,
                                                    OUTPUT / "rankings", branch,
                                                    {**binding, "split": kind, "holdout_sha256": holdout["sha256"]})
            print(json.dumps({"event": "ranking_done", "branch": branch,
                              "hit32": summary["hit@32"]}), flush=True)
        atomic_path = CONFIRM / "data/confirm_atomic.jsonl"
        atomic_rows = read_jsonl(atomic_path)
        branch = f"k{entry['k']}_s{entry['subset']}_seed{seed}_{arm}_atomic"
        eval_lib.evaluate_atomic(model, tokenizer, token_ids, atomic_rows,
                                 OUTPUT / "rankings", branch,
                                 {**binding, "split": "confirm_atomic", "atomic_data_sha256": sha256(atomic_path)})
        print(json.dumps({"event": "atomic_done", "branch": branch}), flush=True)
    finally:
        eval_lib.unload(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--subset", type=int, default=1)
    parser.add_argument("--seed", type=int, default=85000)
    parser.add_argument("--arm", choices=("random", "withheld"), default="random")
    parser.add_argument("--export", type=Path, default=LEGACY / "atomic_export/frozen_atomic_0p6b")
    args = parser.parse_args()
    identity = f"k{args.k}_s{args.subset}_seed{args.seed}_{args.arm}_{args.mode}"
    try:
        with run_lease():
            freeze = verify_freeze()
            entry = entry_for(freeze, args.k, args.subset)
            if args.seed not in freeze["seeds"]:
                raise RuntimeError("training seed not in frozen design")
            core, model_lib, eval_lib = load_libraries(freeze)
            if model_lib._export_payload_sha(args.export).upper() != freeze["atomic_export_payload_sha256"]:
                raise RuntimeError("atomic export payload SHA-256 mismatch")
            if args.mode == "preflight":
                prepared, resolved = prepared_pair(freeze, entry, args.seed, core, model_lib, args.export)
                print(json.dumps({"status": "READY", "job": identity,
                                  "records_each": len(prepared["random"]["encoded"]),
                                  "target_tokens_each": prepared["random"]["target_tokens"],
                                  "dtype": resolved["dtype"]}), flush=True)
            elif args.mode == "train":
                train_one(freeze, entry, args.seed, args.arm, args.export, core, model_lib)
            else:
                evaluate_one(freeze, entry, args.seed, args.arm, args.export, model_lib, eval_lib)
            print(json.dumps({"status": "DONE", "job": identity}), flush=True)
    except Exception as exc:
        failure = {"schema": "withheld-pairs.job-failure.v1", "status": "FAILED", "job": identity,
                   "time_unix": time.time(), "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc()}
        path = OUTPUT / "failures" / identity / f"{time.time_ns()}.json"
        write_exclusive_or_verify(path, json_bytes(failure))
        raise


if __name__ == "__main__":
    main()

"""Resumable, receipt-bound local RTX 5060 Ti trajectory-selection experiment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8"
CONFIRM = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
ATOMIC = REPO / "artifacts/stage4_sh1_v5_0p6b"
OUTPUT = REPO / "artifacts/trajectory_diversity_20260922"
EXPORT = LEGACY / "atomic_export/frozen_atomic_0p6b"
SOURCE = LEGACY / "data/discover_trajectories.jsonl"
sys.path.insert(0, str(LEGACY / "code"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def hash_value(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest().upper()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def validate_training_pair(left: dict, right: dict) -> dict:
    if left.get("status") != right.get("status") or left.get("status") != "DONE":
        raise ValueError("both training receipts must be DONE")
    keys = ("optimizer_steps", "effective_batch", "epochs", "records", "loss_bearing_target_tokens_seen")
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError("training budget differs between arms")
    if not math.isfinite(float(left["mean_loss"])) or not math.isfinite(float(right["mean_loss"])):
        raise ValueError("non-finite training loss")
    return {key: left[key] for key in keys}


def require_freeze() -> dict:
    path = OUTPUT / "SELECTION_FROZEN.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or freeze.get("schema") != "verifier-bottleneck.trajectory-selection-freeze.v1":
        raise RuntimeError("selection freeze invalid")
    if sha256(SOURCE) != freeze["source_sha256"]:
        raise RuntimeError("training source changed after freeze")
    for relative, expected in freeze["code_sha256"].items():
        if sha256(REPO / relative) != expected:
            raise RuntimeError(f"code changed after freeze: {relative}")
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
                raise RuntimeError("another experiment orchestrator holds the run lease") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another experiment orchestrator holds the run lease") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def libs(freeze: dict):
    import torch
    import composition_core as core
    import composition_model as model_lib
    import composition_eval as eval_lib
    if not torch.cuda.is_available() or "RTX 5060 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("the frozen local RTX 5060 Ti is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is unavailable on the local GPU")
    model_lib.ROOT = OUTPUT
    model_lib.PROTOCOL = {"model": "Qwen/Qwen3-0.6B", "pilot_initial": freeze["training_config"]}
    return core, model_lib, eval_lib


def balanced_atomic_pool(core, seed: int) -> list[dict]:
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


def training_pair(seed: int, freeze: dict, core, model_lib) -> dict:
    selection = freeze["selections"][str(seed)]
    source = {row["trajectory_id"]: row for row in read_jsonl(SOURCE)}
    atomic_pool = balanced_atomic_pool(core, seed)
    tokenizer, _ = model_lib.tokenizer_and_ids(EXPORT)
    resolved = {**freeze["training_config"], "schema": "trajectory-diversity.resolved-training.v1",
                "seed": seed, "replicate_label": seed - 83000}
    prepared = {}
    for arm in ("random", "diverse"):
        ids = selection[f"{arm}_ids"]
        trajectories = [source[identity] for identity in ids]
        specs = model_lib.distill_example_specs(trajectories, atomic_pool,
                                                 resolved["atomic_replay"], seed)
        encoded = model_lib.encode_specs(tokenizer, specs)
        expected = selection["budget"]["target_tokens_per_epoch_each"]
        atomic_count = round(len(ids) * resolved["atomic_replay"] / (1 - resolved["atomic_replay"]))
        observed = sum(row["target_tokens"] for row in encoded)
        if len(encoded) != len(ids) + atomic_count:
            raise RuntimeError(f"seed {seed} {arm}: record count mismatch")
        prepared[arm] = {"encoded": encoded, "target_tokens": observed, "hash": hash_value(encoded)}
        if observed < expected:
            raise RuntimeError(f"seed {seed} {arm}: composition target token count impossible")
    if prepared["random"]["target_tokens"] != prepared["diverse"]["target_tokens"]:
        raise RuntimeError(f"seed {seed}: encoded loss-bearing token budgets differ")
    if len(prepared["random"]["encoded"]) != len(prepared["diverse"]["encoded"]):
        raise RuntimeError(f"seed {seed}: encoded record counts differ")
    receipts = {}
    for arm in ("random", "diverse"):
        adapter = OUTPUT / "adapters" / f"seed{seed}_{arm}"
        print(json.dumps({"milestone": "training_start", "seed": seed, "arm": arm}), flush=True)
        receipt = model_lib.train_branch(EXPORT, adapter, prepared[arm]["encoded"], resolved, arm,
                                         prepared[arm]["hash"])
        receipts[arm] = receipt
        print(json.dumps({"milestone": "training_done", "seed": seed, "arm": arm,
                          "optimizer_steps": receipt["optimizer_steps"], "mean_loss": receipt["mean_loss"]}), flush=True)
    equal = validate_training_pair(receipts["random"], receipts["diverse"])
    expected_steps = math.ceil(len(prepared["random"]["encoded"]) / resolved["effective_batch"]) * resolved["epochs"]
    if equal["optimizer_steps"] != expected_steps:
        raise RuntimeError(f"seed {seed}: optimizer steps differ from expected")
    pair = {"schema": "trajectory-diversity.training-pair.v1", "status": "DONE", "seed": seed,
            "budget": equal, "freeze_sha256": sha256(OUTPUT / "SELECTION_FROZEN.json"),
            "adapters": {arm: {"path": str(OUTPUT / "adapters" / f"seed{seed}_{arm}"),
                               "tree_sha256": model_lib.tree_sha256(OUTPUT / "adapters" / f"seed{seed}_{arm}"),
                               "receipt_sha256": sha256(OUTPUT / "adapters" / f"seed{seed}_{arm}" / "training_receipt.json")}
                         for arm in ("random", "diverse")}}
    path = OUTPUT / "runs" / f"seed{seed}_pair.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != pair:
            raise RuntimeError(f"seed {seed}: existing pair receipt differs")
    else:
        write_exclusive(path, pair)
    return pair


def require_training_done(freeze: dict) -> dict[int, dict]:
    pairs = {}
    for seed in range(83000, 83006):
        path = OUTPUT / "runs" / f"seed{seed}_pair.json"
        pair = json.loads(path.read_text(encoding="utf-8"))
        if pair.get("status") != "DONE" or pair.get("freeze_sha256") != sha256(OUTPUT / "SELECTION_FROZEN.json"):
            raise RuntimeError(f"seed {seed}: pair receipt invalid")
        for arm in ("random", "diverse"):
            adapter = OUTPUT / "adapters" / f"seed{seed}_{arm}"
            if sha256(adapter / "training_receipt.json") != pair["adapters"][arm]["receipt_sha256"]:
                raise RuntimeError(f"seed {seed} {arm}: training receipt changed")
        pairs[seed] = pair
    return pairs


def atomic_reference(core):
    data_manifest = json.loads((ATOMIC / "manifests/data_manifest.json").read_text(encoding="utf-8"))
    corrective = json.loads((ATOMIC / "manifests/corrective_data_manifest.json").read_text(encoding="utf-8"))
    manifest = {**data_manifest, **corrective}
    names = ("coordinate_source", "full_sh1", "control_apply", "calibration",
             "corrective_sh1", "corrective_control")
    raw = {name: read_jsonl(ATOMIC / "data" / f"{name}.jsonl") for name in names}
    return core.audit_atomic_reference(manifest, raw, required_splits=("calibration",), strict=True)


def generate_holdout(freeze: dict, core) -> dict:
    require_training_done(freeze)
    holdout_path = OUTPUT / "data/holdout_final_a.jsonl"
    receipt_path = OUTPUT / "data/holdout_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "DONE" or sha256(holdout_path) != receipt["sha256"]:
            raise RuntimeError("existing holdout receipt invalid")
        return receipt
    if holdout_path.exists():
        raise RuntimeError("holdout exists without validated receipt; preserve it for audit")
    reference = atomic_reference(core)
    registry = core.FingerprintRegistry.from_atomic_reference(reference)
    existing = {name: read_jsonl(LEGACY / "data" / f"{name}.jsonl") for name in ("train", "dev_a", "dev_b")}
    existing["source_trajectories"] = read_jsonl(SOURCE)
    for path in sorted((CONFIRM / "data").glob("final_*.jsonl")):
        existing[path.stem] = read_jsonl(path)
    core._reserve_existing_rows(registry, existing)
    forbidden_states = set(registry.states)
    forbidden_tasks = set(registry.tasks)
    rows = core.generate_split("final_a", 1000, "A", seed=freeze["holdout_seed"],
                               depths=(3,), registry=registry)
    state_sets = []
    for row in rows:
        errors, states, task = core._validate_task_row(row, "final_a", verify_shortest=True)
        if errors or task in forbidden_tasks or states & forbidden_states:
            raise RuntimeError(f"holdout task invalid or overlapping: {row['task_id']} {errors}")
        state_sets.append(states)
    if len({row["task_id"] for row in rows}) != 1000:
        raise RuntimeError("holdout task ids are not unique")
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with holdout_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    receipt = {"schema": "trajectory-diversity.holdout.v1", "status": "DONE", "rows": len(rows),
               "sha256": sha256(holdout_path), "seed": freeze["holdout_seed"],
               "forbidden_task_count": len(forbidden_tasks), "forbidden_state_count": len(forbidden_states),
               "source_freeze_sha256": sha256(OUTPUT / "SELECTION_FROZEN.json")}
    write_exclusive(receipt_path, receipt)
    print(json.dumps({"milestone": "holdout_frozen", "rows": len(rows), "sha256": receipt["sha256"]}), flush=True)
    return receipt


def evaluate_all(freeze: dict, model_lib, eval_lib, *, atomic: bool) -> None:
    pairs = require_training_done(freeze)
    holdout = generate_holdout(freeze, __import__("composition_core"))
    rows = read_jsonl(OUTPUT / "data/holdout_final_a.jsonl")
    atomic_rows = read_jsonl(CONFIRM / "data/confirm_atomic.jsonl") if atomic else None
    for seed in range(83000, 83006):
        for arm in ("random", "diverse"):
            adapter = OUTPUT / "adapters" / f"seed{seed}_{arm}"
            binding = {"schema": "trajectory-diversity.eval-binding.v1", "seed": seed, "arm": arm,
                       "holdout_sha256": holdout["sha256"],
                       "adapter_tree_sha256": pairs[seed]["adapters"][arm]["tree_sha256"],
                       "selection_freeze_sha256": sha256(OUTPUT / "SELECTION_FROZEN.json")}
            model, tokenizer, token_ids = model_lib.load_branch(EXPORT, adapter)
            try:
                branch = f"seed{seed}_{arm}_holdout"
                summary, _ = eval_lib.evaluate_ranking(model, tokenizer, token_ids, rows,
                                                       OUTPUT / "rankings", branch, binding)
                print(json.dumps({"milestone": "evaluation_done", "seed": seed, "arm": arm,
                                  "hit32": summary["hit@32"]}), flush=True)
                if atomic:
                    atomic_binding = {**binding, "split": "confirm_atomic",
                                      "atomic_data_sha256": sha256(CONFIRM / "data/confirm_atomic.jsonl")}
                    eval_lib.evaluate_atomic(model, tokenizer, token_ids, atomic_rows,
                                             OUTPUT / "rankings", f"seed{seed}_{arm}_atomic", atomic_binding)
                    print(json.dumps({"milestone": "atomic_evaluation_done", "seed": seed, "arm": arm}), flush=True)
            finally:
                eval_lib.unload(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "holdout", "evaluate", "all"), required=True)
    args = parser.parse_args()
    with run_lease():
        freeze = require_freeze()
        core, model_lib, eval_lib = libs(freeze)
        if args.mode in ("train", "all"):
            for seed in range(83000, 83006):
                training_pair(seed, freeze, core, model_lib)
        if args.mode in ("holdout", "all"):
            generate_holdout(freeze, core)
        if args.mode in ("evaluate", "all"):
            evaluate_all(freeze, model_lib, eval_lib, atomic=True)
        print(json.dumps({"status": "DONE", "mode": args.mode}), flush=True)


if __name__ == "__main__":
    main()

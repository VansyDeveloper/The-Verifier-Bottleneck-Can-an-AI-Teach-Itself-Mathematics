from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
V1 = REPO / "artifacts/stage4_distill_v1/code"
V2 = REPO / "artifacts/stage4_distill_v2/code"
V3 = REPO / "artifacts/stage4_distill_v3/code"
sys.path[:0] = [str(V3), str(V2), str(V1)]

import v2_atomic as v2
import v3_train as v3
from stage4_core import OPS, TOKENS, apply_prompt, dump_json, format_state, plan_prompt, read_jsonl, sha256, trajectory, write_jsonl

CFG = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
DATA, RUNS, ADAPTERS, MANIFESTS = (ROOT / name for name in ("data", "runs", "adapters", "manifests"))
MODE_QUOTA = ("dense",) * 4 + ("boundary",) * 4 + ("sparse",) * 3 + ("monomial",) * 3 + ("progression",) * 3 + ("alternating",) * 3


def dump_json_exclusive(path, value):
    """Create an ownership marker with O_EXCL; never overwrite a peer claim."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())


def dump_json_atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.partial")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    partial.replace(path)


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # ``os.kill(pid, 0)`` terminates a process on Windows.  A heartbeat
        # must use the read-only process-query API instead.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def state_key(p, state):
    return int(p), tuple(state)


def state_key_digest(p, state):
    raw = json.dumps([int(p), list(state)], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def tree_sha256(path):
    path = Path(path)
    if not path.is_dir():
        raise RuntimeError(f"cannot fingerprint missing directory: {path}")
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"cannot fingerprint empty directory: {path}")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(file_path)))
    return digest.hexdigest()


def frozen_dataset_sha(split):
    manifest = json.loads((MANIFESTS / "data_manifest.json").read_text(encoding="utf-8"))
    return manifest[f"{split}.jsonl"]["sha256"]


def verify_frozen_dataset(split):
    actual = sha256(DATA / f"{split}.jsonl")
    expected = frozen_dataset_sha(split)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"frozen {split} hash mismatch")
    return actual


def gate_binding(adapter, split, run_id):
    return {
        "schema": "stage4.sh1.v5.gate-binding.v1",
        "run_id": run_id,
        "split": split,
        "model": CFG["model"],
        "protocol_sha256": sha256(ROOT / "configs/protocol.json"),
        "dataset_sha256": frozen_dataset_sha(split),
        "adapter_tree_sha256": tree_sha256(adapter),
        "evaluator_sha256": sha256(Path(__file__)),
    }


def load_bound_gate(path, adapter, split, run_id):
    path = Path(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = gate_binding(adapter, split, run_id)
    if gate.get("binding") != expected:
        raise RuntimeError(f"stale or mismatched cached gate: {path}")
    generation_path = RUNS / "generations" / f"{run_id}.jsonl"
    if not generation_path.exists() or gate.get("generation_sha256", "").lower() != sha256(generation_path).lower():
        raise RuntimeError(f"generation hash mismatch for cached gate: {path}")
    if gate.get("generation_rows") != len(read_jsonl(generation_path)):
        raise RuntimeError(f"generation row-count mismatch for cached gate: {path}")
    return gate


def task_id(split, op, p, state, nonce=0):
    raw = json.dumps([split, op, p, list(state), nonce], separators=(",", ":"))
    return "s4v5-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def prior_evaluation_states():
    paths = (
        REPO / "artifacts/stage4_distill_v2/data/atomic_dev.jsonl",
        REPO / "artifacts/stage4_distill_v3/data/fresh_atomic_test.jsonl",
        REPO / "artifacts/stage4_sh1_v4/data/fresh_final_test.jsonl",
    )
    out = set()
    for path in paths:
        out.update(state_key(r["p"], r["start"]) for r in read_jsonl(path))
    return out


def make_unique_state(rng, p, degree, mode, forbidden, attempts=5000):
    for at in range(attempts):
        chosen = mode if at < 200 else rng.choice(CFG["modes"])
        state = v3.make_state(rng, p, degree, chosen)
        if any(state) and state_key(p, state) not in forbidden:
            return state, chosen
    raise RuntimeError(f"state generation exhausted p={p} degree={degree} mode={mode}")


def make_unique_state_hashed(rng, p, degree, mode, forbidden_hashes, attempts=5000):
    for at in range(attempts):
        chosen = mode if at < 200 else rng.choice(CFG["modes"])
        state = v3.make_state(rng, p, degree, chosen)
        if any(state) and state_key_digest(p, state) not in forbidden_hashes:
            return state, chosen
    raise RuntimeError(f"hashed state generation exhausted p={p} degree={degree} mode={mode}")


def make_corrective_state(rng, p, degree, mode, unique_forbidden_hashes, eval_forbidden_hashes):
    """Prefer a new state, then relax uniqueness only after that search is exhausted."""
    try:
        state, actual_mode = make_unique_state_hashed(rng, p, degree, mode, unique_forbidden_hashes)
        unique_forbidden_hashes.add(state_key_digest(p, state))
        return state, actual_mode, False
    except RuntimeError:
        state, actual_mode = make_unique_state_hashed(rng, p, degree, mode, eval_forbidden_hashes)
        digest = state_key_digest(p, state)
        reused = digest in unique_forbidden_hashes
        unique_forbidden_hashes.add(digest)
        return state, actual_mode, reused


def balanced_eval(split, seed, forbidden):
    rng = random.Random(seed)
    rows, states = [], []
    for p in CFG["fields"]:
        for degree in CFG["degrees"]:
            modes = list(MODE_QUOTA)
            rng.shuffle(modes)
            for mode in modes:
                state, actual_mode = make_unique_state(rng, p, degree, mode, forbidden)
                forbidden.add(state_key(p, state))
                states.append((p, degree, actual_mode, state))
    for p, degree, mode, state in states:
        for op in OPS:
            target = trajectory(state, [op], p)[-1]
            rows.append({"schema": "stage4.sh1.v5.task.v1", "task_id": task_id(split, op, p, state), "split": split,
                         "p": p, "degree": degree, "state_mode": mode, "start": list(state), "operation": op,
                         "program": [op], "witness": [op], "target": list(target)})
    rng.shuffle(rows)
    return rows


def allocate(total, cells):
    base, rem = divmod(total, len(cells))
    return {cell: base + (i < rem) for i, cell in enumerate(cells)}


def generate_sh1_states(split, by_degree, seed, forbidden):
    rng = random.Random(seed)
    rows, seen = [], set()
    for degree in CFG["degrees"]:
        total = int(by_degree[str(degree)])
        cells = [(p, mode) for p in CFG["fields"] for mode in CFG["modes"]]
        counts = allocate(total, cells)
        for p, mode in cells:
            for presentation in range(counts[(p, mode)]):
                state = actual_mode = None
                for at in range(500):
                    chosen = mode if at < 200 else rng.choice(CFG["modes"])
                    candidate = v3.make_state(rng, p, degree, chosen)
                    candidate_key = state_key(p, candidate)
                    if any(candidate) and candidate_key not in forbidden and candidate_key not in seen:
                        state, actual_mode = candidate, chosen; break
                if state is None:
                    # Finite cells may exhaust unique train states; evaluation states remain hard-forbidden.
                    for at in range(500):
                        chosen = mode if at < 200 else rng.choice(CFG["modes"])
                        candidate = v3.make_state(rng, p, degree, chosen)
                        if any(candidate) and state_key(p, candidate) not in forbidden:
                            state, actual_mode = candidate, chosen; break
                if state is None: raise RuntimeError(f"training state generation exhausted p={p} degree={degree}")
                seen.add(state_key(p, state))
                target = trajectory(state, ["SH1"], p)[-1]
                rows.append({"schema": "stage4.sh1.v5.task.v1", "task_id": task_id(split, "SH1", p, state, len(rows)), "split": split,
                             "p": p, "degree": degree, "state_mode": actual_mode, "start": list(state), "operation": "SH1",
                             "program": ["SH1"], "witness": ["SH1"], "target": list(target)})
    rng.shuffle(rows)
    return rows


def operation_row(source, op, split, nonce=0):
    target = trajectory(source["start"], [op], source["p"])[-1]
    return {**source, "task_id": task_id(split, op, source["p"], source["start"], nonce), "split": split,
            "operation": op, "program": [op], "witness": [op], "target": list(target)}


def prepare():
    for d in (DATA, RUNS, ADAPTERS, MANIFESTS): d.mkdir(parents=True, exist_ok=True)
    eval_forbidden = prior_evaluation_states()
    calibration = balanced_eval("calibration", 75001, eval_forbidden)
    final = balanced_eval("final", 75002, eval_forbidden)
    train_forbidden = {state_key(r["p"], r["start"]) for r in calibration + final}
    coordinate = generate_sh1_states("coordinate_source", CFG["coordinate_states_by_degree"], 75003, train_forbidden)
    full = generate_sh1_states("full_sh1", CFG["full_sh1_by_degree"], 75004, train_forbidden)
    rng = random.Random(75005)
    controls = []
    for op in ("SC2", "REV", "AC1", "AX1"):
        for i, source in enumerate(rng.sample(full, CFG["control_apply_per_operation"])):
            controls.append(operation_row(source, op, "control_apply", i))
    for name, rows in (("calibration", calibration), ("final", final), ("coordinate_source", coordinate),
                       ("full_sh1", full), ("control_apply", controls)):
        write_jsonl(DATA / f"{name}.jsonl", rows)
    split_states = {name: {state_key(r["p"], r["start"]) for r in rows} for name, rows in
                    (("calibration", calibration), ("final", final), ("coordinate", coordinate), ("full", full))}
    audit = {"status": "PASS", "counts": {"calibration": len(calibration), "final": len(final), "coordinate": len(coordinate),
             "full_sh1": len(full), "control_apply": len(controls)},
             "train_unique_states": {"coordinate": len(split_states["coordinate"]), "full": len(split_states["full"])},
             "train_repeated_presentations": {"coordinate": len(coordinate) - len(split_states["coordinate"]),
                                                "full": len(full) - len(split_states["full"])},
             "calibration_by_field": Counter(r["p"] for r in calibration), "final_by_field": Counter(r["p"] for r in final),
             "calibration_by_degree": Counter(r["degree"] for r in calibration), "final_by_degree": Counter(r["degree"] for r in final),
             "calibration_final_overlap": len(split_states["calibration"] & split_states["final"]),
             "train_eval_overlap": len((split_states["coordinate"] | split_states["full"]) &
                                       (split_states["calibration"] | split_states["final"]))}
    assert audit["calibration_final_overlap"] == audit["train_eval_overlap"] == 0
    assert all(v == 300 for v in audit["calibration_by_field"].values()) and all(v == 600 for v in audit["calibration_by_degree"].values())
    assert all(v == 300 for v in audit["final_by_field"].values()) and all(v == 600 for v in audit["final_by_degree"].values())
    dump_json(MANIFESTS / "data_audit.json", audit)
    dump_json(MANIFESTS / "data_manifest.json", {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size,
              "rows": len(read_jsonl(p))} for p in DATA.glob("*.jsonl")})
    seal_eval_exclusion(calibration, final)
    dump_json(MANIFESTS / "environment.json", {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
              "model": CFG["model"], "parent_adapter": CFG["parent_adapter"], "seed": CFG["seed"],
              "torch": v2.torch.__version__, "cuda": v2.torch.version.cuda,
              "gpu": v2.torch.cuda.get_device_name(0) if v2.torch.cuda.is_available() else None})
    print(json.dumps(audit, indent=2, default=dict))


def seal_eval_exclusion(calibration=None, final=None):
    """Seal state digests used only to keep later training disjoint from evaluation."""
    calibration = calibration if calibration is not None else read_jsonl(DATA / "calibration.jsonl")
    final = final if final is not None else read_jsonl(DATA / "final.jsonl")
    keys = sorted({state_key_digest(r["p"], r["start"]) for r in calibration + final})
    payload = {
        "schema": "stage4.sh1.v5.eval-exclusion.v1",
        "purpose": "state-disjointness only; contains no split labels, targets, metrics, or model outputs",
        "calibration_sha256": sha256(DATA / "calibration.jsonl"),
        "final_sha256": sha256(DATA / "final.jsonl"),
        "calibration_rows": len(calibration),
        "final_rows": len(final),
        "unique_state_digests": len(keys),
        "state_digests": keys,
    }
    dump_json(MANIFESTS / "eval_state_exclusion.json", payload)
    return payload


def coordinate_prompt(r, j):
    terms = " + ".join(f"c{i}*{math.comb(i, j)}" for i in range(j, r["degree"] + 1))
    return (f"FIELD: {r['p']}\nDEGREE_CAP: {r['degree']}\nSTART: {format_state(r['start'])}\nPROGRAM: <OP0>\n"
            f"OUTPUT_COORDINATE: {j}\nRULE: b{j} = ({terms}) modulo FIELD.\nReturn exactly the integer:\nVALUE:")


def prefix_prompt(r, length):
    return (f"FIELD: {r['p']}\nDEGREE_CAP: {r['degree']}\nSTART: {format_state(r['start'])}\nPROGRAM: <OP0>\n"
            f"OUTPUT_PREFIX_LENGTH: {length}\nRULE: b_j = sum over i>=j of c_i*binomial(i,j), modulo FIELD.\n"
            "Return exactly the coefficient prefix:\nRESULT:")


def configure_v2():
    v2.RUNS, v2.DATA, v2.ADAPTERS, v2.MANIFESTS = RUNS, DATA, ADAPTERS, MANIFESTS


def save_adapter(model, path, receipt):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite adapter output: {path}")
    partial.mkdir(parents=True)
    model.save_pretrained(partial); v2.TOKENIZER.save_pretrained(partial); dump_json(partial / "training_receipt.json", receipt)
    partial.replace(path)


def phase_input_hashes(phase):
    names = {
        "coordinate": ("coordinate_source.jsonl",),
        "prefix": ("coordinate_source.jsonl",),
        "full": ("full_sh1.jsonl", "control_apply.jsonl"),
        "corrective": ("corrective_sh1.jsonl", "corrective_control.jsonl"),
    }[phase]
    return {name: sha256(DATA / name) for name in names}


def expected_phase_binding(parent, output, spec, phase, examples):
    return {
        "schema": "stage4.sh1.v5.phase-seal.v1",
        "phase": phase,
        "model": CFG["model"],
        "examples": examples,
        "spec": spec,
        "protocol_sha256": sha256(ROOT / "configs/protocol.json"),
        "input_sha256": phase_input_hashes(phase),
        "parent_tree_sha256": tree_sha256(parent),
        "output_tree_sha256": tree_sha256(output),
        "receipt_sha256": sha256(Path(output) / "training_receipt.json"),
    }


def validate_phase_receipt(parent, output, spec, phase, examples):
    output = Path(output); receipt_path = output / "training_receipt.json"
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"partial adapter output requires audit before resume: {partial}")
    if not receipt_path.exists():
        if output.exists():
            raise RuntimeError(f"partial adapter without DONE receipt: {output}")
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inner = receipt.get("phase", {})
    expected_parent = str(Path(parent))
    checks = {
        "outer_status": receipt.get("status") == "DONE",
        "parent": receipt.get("parent") == expected_parent,
        "phase": inner.get("phase") == phase,
        "inner_status": inner.get("status") == "DONE",
        "examples": inner.get("examples") == examples,
        "epochs": inner.get("epochs") == spec["epochs"],
        "lr": inner.get("lr") == spec["lr"],
        "effective_batch": inner.get("effective_batch") == spec["effective_batch"],
        "adapter_config": (output / "adapter_config.json").is_file(),
        "adapter_weights": (output / "adapter_model.safetensors").is_file() or (output / "adapter_model.bin").is_file(),
        "tokenizer": (output / "tokenizer.json").is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid phase receipt {phase}: {failed}")
    return True


def phase_seal_path(phase):
    return MANIFESTS / "phase_seals" / f"{phase}.json"


def seal_phase(parent, output, spec, phase, examples):
    if not validate_phase_receipt(parent, output, spec, phase, examples):
        raise RuntimeError(f"cannot seal incomplete phase: {phase}")
    payload = expected_phase_binding(parent, output, spec, phase, examples)
    environment = json.loads((MANIFESTS / "environment.json").read_text(encoding="utf-8"))
    payload["training_launch_git_commit"] = environment["git_commit"] if phase in ("coordinate", "prefix", "full") else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dump_json(phase_seal_path(phase), payload)
    return payload


def phase_complete(parent, output, spec, phase, examples):
    if not validate_phase_receipt(parent, output, spec, phase, examples):
        return False
    seal_path = phase_seal_path(phase)
    if not seal_path.exists():
        raise RuntimeError(f"completed {phase} is not hash-sealed; run seal-training before resume")
    actual = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = expected_phase_binding(parent, output, spec, phase, examples)
    if any(actual.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"stale or mismatched phase seal: {phase}")
    return True


def seal_training():
    phases = (
        (REPO / CFG["parent_adapter"], ADAPTERS / "coordinate", CFG["phase_coordinate"], "coordinate", 104000),
        (ADAPTERS / "coordinate", ADAPTERS / "prefix", CFG["phase_prefix"], "prefix", 80000),
        (ADAPTERS / "prefix", ADAPTERS / "curriculum", CFG["phase_full"], "full", 90000),
        (ADAPTERS / "curriculum", ADAPTERS / "corrective", CFG["phase_corrective"], "corrective", 55000),
    )
    sealed = {}
    for parent, output, spec, phase, examples in phases:
        if (output / "training_receipt.json").exists():
            sealed[phase] = seal_phase(parent, output, spec, phase, examples)
        else:
            break
    return sealed


def exact_apply_metrics(generations):
    by = defaultdict(lambda: [0, 0])
    correct = 0
    for row in generations:
        legacy = bool(row.get("correct"))
        exact = row["raw_completion"].strip() == row["target"]
        row["legacy_prefix_correct"] = legacy
        row["correct"] = bool(exact)
        row["match_mode"] = "stripped_exact_equality"
        op = row["operation"]
        correct += exact
        by[op][0] += exact
        by[op][1] += 1
    return {"overall": correct / len(generations),
            "by_operation": {op: passed / total for op, (passed, total) in by.items()},
            "match_mode": "stripped_exact_equality"}


def run_training_phase(parent, output, encoded, spec, phase):
    configure_v2()
    status_path = RUNS / f"{phase}_status.json"
    status = {"schema": "stage4.sh1.v5.training-status.v1", "status": "STARTED", "phase": phase,
              "parent": str(parent), "output": str(output), "examples": len(encoded),
              "model": CFG["model"], "protocol_sha256": sha256(ROOT / "configs/protocol.json"),
              "input_sha256": phase_input_hashes(phase), "code_sha256": sha256(Path(__file__)),
              "started_at_unix": time.time()}
    dump_json(status_path, status)
    model = None
    try:
        model, _ = v2.load_continuation(parent)
        receipt = v2.train_phase(model, encoded, spec, phase)
        save_adapter(model, output, {"status": "DONE", "parent": str(parent), "phase": receipt})
        status.update({"status": "DONE", "finished_at_unix": time.time(), "adapter_tree_sha256": tree_sha256(output),
                       "training_receipt_sha256": sha256(Path(output) / "training_receipt.json")})
        dump_json(status_path, status)
        seal_phase(parent, output, spec, phase, len(encoded))
        return receipt
    except Exception as exc:
        status.update({"status": "FAILED", "finished_at_unix": time.time(), "error_type": type(exc).__name__, "error": str(exc)})
        dump_json(status_path, status)
        raise
    finally:
        if model is not None: del model
        gc.collect()
        if v2.torch.cuda.is_available(): v2.torch.cuda.empty_cache()


def train():
    coordinate_rows = read_jsonl(DATA / "coordinate_source.jsonl")
    coordinate_adapter = ADAPTERS / "coordinate"
    if not phase_complete(REPO / CFG["parent_adapter"], coordinate_adapter, CFG["phase_coordinate"], "coordinate", 104000):
        encoded = [v2.encode_pair(v2.TOKENIZER, coordinate_prompt(r, j), str(value))
                   for r in coordinate_rows for j, value in enumerate(r["target"])]
        random.Random(75010).shuffle(encoded)
        run_training_phase(REPO / CFG["parent_adapter"], coordinate_adapter, encoded, CFG["phase_coordinate"], "coordinate")
    prefix_adapter = ADAPTERS / "prefix"
    if not phase_complete(coordinate_adapter, prefix_adapter, CFG["phase_prefix"], "prefix", 80000):
        encoded = [v2.encode_pair(v2.TOKENIZER, prefix_prompt(r, length), format_state(r["target"][:length]))
                   for r in coordinate_rows for length in range(2, r["degree"] + 2)]
        assert len(encoded) == 80000
        random.Random(75011).shuffle(encoded)
        run_training_phase(coordinate_adapter, prefix_adapter, encoded, CFG["phase_prefix"], "prefix")
    curriculum = ADAPTERS / "curriculum"
    if not phase_complete(prefix_adapter, curriculum, CFG["phase_full"], "full", 90000):
        full = read_jsonl(DATA / "full_sh1.jsonl") + read_jsonl(DATA / "control_apply.jsonl")
        encoded = [v2.encode_pair(v2.TOKENIZER, apply_prompt(r), format_state(r["target"])) for r in full]
        rng = random.Random(75012); per_op = CFG["plan_replay"] // len(OPS)
        for op in OPS:
            candidates = [r for r in full if r["operation"] == op]
            for r in rng.choices(candidates, k=per_op):
                encoded.append(v2.encode_pair(v2.TOKENIZER, plan_prompt({**r, "depth": 1}), TOKENS[op]))
        assert len(encoded) == 90000
        rng.shuffle(encoded)
        run_training_phase(prefix_adapter, curriculum, encoded, CFG["phase_full"], "full")


def evaluate_adapter(adapter, split, run_id):
    configure_v2()
    access_receipt = authorize_final_access(adapter, run_id) if split == "final" else None
    verify_frozen_dataset(split)
    rows = read_jsonl(DATA / f"{split}.jsonl"); model, _ = v2.load_continuation(adapter)
    gc.collect()
    if v2.torch.cuda.is_available(): v2.torch.cuda.empty_cache()
    model.float(); model.eval(); started = time.time()
    plan = v2.atomic_plan_accuracy(model, v2.TOKENIZER, v2.TIDS, rows, batch_size=16)
    _, generations = v2.atomic_apply_accuracy(model, v2.TOKENIZER, rows, batch_size=8, return_generations=True)
    apply = exact_apply_metrics(generations)
    gate = v2.atomic_gate(plan, apply)
    control_checks = {op: apply["by_operation"][op] >= CFG["v4_control_baseline"][op] - CFG["gates"]["control_drop"]
                      for op in ("SC2", "REV", "AC1", "AX1")}
    gate["checks"]["control_forgetting_le_002"] = all(control_checks.values())
    gate["control_checks"] = control_checks; gate["pass"] = all(gate["checks"].values())
    gate.update({"run_id": run_id, "split": split, "evaluation_dtype": "float32", "wall_seconds": time.time() - started,
                 "binding": gate_binding(adapter, split, run_id)})
    generation_path = RUNS / "generations" / f"{run_id}.jsonl"
    gate_path = RUNS / f"{run_id}_gate.json"
    write_jsonl(generation_path, generations)
    gate.update({"generation_sha256": sha256(generation_path), "generation_rows": len(generations)})
    dump_json(gate_path, gate)
    if access_receipt is not None:
        finalize_final_access(adapter, run_id, gate_path)
    del model; gc.collect()
    if v2.torch.cuda.is_available(): v2.torch.cuda.empty_cache()
    return gate


def calibration_gate_for(adapter):
    name = Path(adapter).name
    if name == "curriculum":
        return RUNS / "curriculum_calibration_gate.json"
    if name == "corrective":
        return RUNS / "corrective_calibration_gate.json"
    raise RuntimeError(f"final access is not authorized for adapter {adapter}")


def authorize_final_access(adapter, run_id):
    gate_path = calibration_gate_for(adapter)
    if not gate_path.exists():
        raise RuntimeError("frozen final remains closed: calibration gate is missing")
    calibration_run_id = "curriculum_calibration" if Path(adapter).name == "curriculum" else "corrective_calibration"
    gate = load_bound_gate(gate_path, adapter, "calibration", calibration_run_id)
    if not gate.get("pass"):
        raise RuntimeError("frozen final remains closed: calibration gate did not pass")
    exclusion_path = MANIFESTS / "eval_state_exclusion.json"
    if not exclusion_path.exists():
        raise RuntimeError("frozen final remains closed: evaluation exclusion seal is missing")
    receipt_path = MANIFESTS / "final_access_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "DONE" and (RUNS / f"{run_id}_gate.json").exists():
            raise RuntimeError("frozen final was already evaluated; load the existing gate instead")
        expected = {"run_id": run_id, "adapter": str(adapter), "calibration_gate": str(gate_path),
                    "calibration_gate_sha256": sha256(gate_path), "eval_exclusion_sha256": sha256(exclusion_path)}
        if receipt.get("status") != "STARTED" or any(receipt.get(key) != value for key, value in expected.items()):
            raise RuntimeError("frozen final access receipt is stale or belongs to another evaluation")
        if (RUNS / f"{run_id}_gate.json").exists():
            raise RuntimeError("a final gate exists but the access receipt is not DONE")
        owner_pid = receipt.get("owner_pid")
        if owner_pid != os.getpid() and pid_alive(owner_pid):
            raise RuntimeError(f"frozen final evaluation is already active in PID {owner_pid}")
        receipt["owner_pid"] = os.getpid()
        receipt["resume_count"] = int(receipt.get("resume_count", 0)) + 1
        receipt["resumed_at_unix"] = time.time()
        dump_json_atomic(receipt_path, receipt)
        return receipt
    receipt = {
        "schema": "stage4.sh1.v5.final-access.v1",
        "status": "STARTED",
        "run_id": run_id,
        "adapter": str(adapter),
        "calibration_gate": str(gate_path),
        "calibration_gate_sha256": sha256(gate_path),
        "eval_exclusion_sha256": sha256(exclusion_path),
        "owner_pid": os.getpid(),
        "resume_count": 0,
        "started_at_unix": time.time(),
    }
    try:
        dump_json_exclusive(receipt_path, receipt)
        return receipt
    except FileExistsError:
        return authorize_final_access(adapter, run_id)


def finalize_final_access(adapter, run_id, gate_path):
    """Recover/complete STARTED -> DONE from already written bound raw outputs."""
    receipt_path = MANIFESTS / "final_access_receipt.json"
    if not receipt_path.is_file(): raise RuntimeError("final gate exists without an access receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    calibration_path = calibration_gate_for(adapter)
    exclusion_path = MANIFESTS / "eval_state_exclusion.json"
    expected = {"run_id": run_id, "adapter": str(adapter), "calibration_gate": str(calibration_path),
                "calibration_gate_sha256": sha256(calibration_path), "eval_exclusion_sha256": sha256(exclusion_path)}
    if receipt.get("status") not in ("STARTED", "DONE") or any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("final access receipt does not bind the completed gate")
    gate = load_bound_gate(gate_path, adapter, "final", run_id)
    generation_path = RUNS / "generations" / f"{run_id}.jsonl"
    done = {**receipt, "status": "DONE", "generation_sha256": sha256(generation_path),
            "gate_sha256": sha256(gate_path), "generation_rows": len(read_jsonl(generation_path))}
    if receipt.get("status") == "DONE":
        for key in ("generation_sha256", "gate_sha256", "generation_rows"):
            if receipt.get(key) != done[key]: raise RuntimeError(f"completed final access {key} mismatch")
        return receipt
    done["finished_at_unix"] = time.time(); dump_json_atomic(receipt_path, done); return done


def corrective_rows(gate_path):
    gate = load_bound_gate(gate_path, ADAPTERS / "curriculum", "calibration", "curriculum_calibration")
    rows = read_jsonl(DATA / "calibration.jsonl")
    generations = read_jsonl(RUNS / "generations/curriculum_calibration.jsonl")
    expected_ids = {r["task_id"] for r in rows}; observed_ids = [g["task_id"] for g in generations]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise RuntimeError("calibration generations are incomplete, duplicated, or mismatched")
    if gate.get("generation_rows") != len(generations):
        raise RuntimeError("calibration gate/generation row mismatch")
    tasks = {r["task_id"]: r for r in rows}; grouped = defaultdict(list)
    for g in generations:
        if g["operation"] == "SH1":
            r = tasks[g["task_id"]]; grouped[(r["p"], r["degree"])].append(bool(g["correct"]))
    cells = [(p, d) for p in CFG["fields"] for d in CFG["degrees"]]
    accuracy = {cell: sum(grouped[cell]) / len(grouped[cell]) for cell in cells}
    raw = {cell: min(CFG["corrective"]["cell_weight_max"], max(CFG["corrective"]["cell_weight_min"],
           1.0 - accuracy[cell])) for cell in cells}
    total_weight = sum(raw.values()); exact = {cell: CFG["corrective"]["sh1"] * raw[cell] / total_weight for cell in cells}
    counts = {cell: int(exact[cell]) for cell in cells}
    for cell in sorted(cells, key=lambda c: exact[c] - counts[c], reverse=True)[:CFG["corrective"]["sh1"] - sum(counts.values())]: counts[cell] += 1
    exclusion_path = MANIFESTS / "eval_state_exclusion.json"
    if not exclusion_path.exists():
        raise RuntimeError("evaluation exclusion seal is required before corrective data generation")
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    eval_digests = set(exclusion["state_digests"])
    forbidden_hashes = set(eval_digests)
    rng = random.Random(75020)
    sh1 = []; reuse_counts = Counter(); unique_counts = Counter()
    for (p, degree), count in counts.items():
        for i in range(count):
            state, mode, reused = make_corrective_state(rng, p, degree, rng.choice(CFG["modes"]),
                                                        forbidden_hashes, eval_digests)
            reuse_counts[(p, degree)] += int(reused); unique_counts[(p, degree)] += int(not reused)
            target = trajectory(state, ["SH1"], p)[-1]
            sh1.append({"schema": "stage4.sh1.v5.task.v1", "task_id": task_id("corrective", "SH1", p, state, i),
                        "split": "corrective", "p": p, "degree": degree, "state_mode": mode, "start": list(state),
                        "operation": "SH1", "program": ["SH1"], "witness": ["SH1"], "target": list(target)})
    base = read_jsonl(DATA / "full_sh1.jsonl"); controls = []
    for op in ("SC2", "REV", "AC1", "AX1"):
        for i, source in enumerate(rng.choices(base, k=CFG["corrective"]["control_apply"] // 4)):
            controls.append(operation_row(source, op, "corrective_control", i))
    write_jsonl(DATA / "corrective_sh1.jsonl", sh1); write_jsonl(DATA / "corrective_control.jsonl", controls)
    corrective_ids = [r["task_id"] for r in sh1 + controls]
    overlap = sum(state_key_digest(r["p"], r["start"]) in eval_digests for r in sh1 + controls)
    corrective_manifest = {
        name: {"sha256": sha256(DATA / name), "bytes": (DATA / name).stat().st_size, "rows": len(read_jsonl(DATA / name))}
        for name in ("corrective_sh1.jsonl", "corrective_control.jsonl")
    }
    dump_json(MANIFESTS / "corrective_data_manifest.json", corrective_manifest)
    dump_json(MANIFESTS / "corrective_allocation.json", {"schema": "stage4.sh1.v5.corrective-allocation.v1",
              "source_gate_sha256": sha256(gate_path), "source_generation_sha256": gate["generation_sha256"],
              "cell_accuracy": {f"{p}:{d}": accuracy[(p, d)] for p, d in cells},
              "cell_weight": {f"{p}:{d}": raw[(p, d)] for p, d in cells}, "cell_count": {f"{p}:{d}": counts[(p, d)] for p, d in cells},
              "unique_state_count": {f"{p}:{d}": unique_counts[(p, d)] for p, d in cells},
              "reused_after_unique_search_exhaustion_count": {f"{p}:{d}": reuse_counts[(p, d)] for p, d in cells},
              "task_ids_unique": len(corrective_ids) == len(set(corrective_ids)), "evaluation_state_overlap": overlap,
              "data": corrective_manifest})
    if len(corrective_ids) != len(set(corrective_ids)) or overlap:
        raise RuntimeError("corrective data integrity failure")
    return sh1, controls


def corrective_allowed(gate):
    return (gate["plan"]["overall"] >= CFG["gates"]["plan"] and
            all(gate["control_checks"].values()) and
            gate["apply"]["by_operation"]["SH1"] < CFG["gates"]["apply_each"])


def corrective():
    initial_gate = json.loads((RUNS / "curriculum_calibration_gate.json").read_text())
    if not corrective_allowed(initial_gate): raise RuntimeError("corrective cycle is not authorized by preregistration")
    sh1, controls = corrective_rows(RUNS / "curriculum_calibration_gate.json")
    encoded = [v2.encode_pair(v2.TOKENIZER, apply_prompt(r), format_state(r["target"])) for r in sh1 + controls]
    rng = random.Random(75021); plan_n = CFG["corrective"]["plan"] // len(OPS)
    all_rows = sh1 + controls
    for op in OPS:
        candidates = [r for r in all_rows if r["operation"] == op]
        for r in rng.choices(candidates, k=plan_n): encoded.append(v2.encode_pair(v2.TOKENIZER, plan_prompt({**r, "depth": 1}), TOKENS[op]))
    assert len(encoded) == 55000; rng.shuffle(encoded)
    run_training_phase(ADAPTERS / "curriculum", ADAPTERS / "corrective", encoded, CFG["phase_corrective"], "corrective")
    seal_phase(ADAPTERS / "curriculum", ADAPTERS / "corrective", CFG["phase_corrective"], "corrective", 55000)


def run():
    if not (DATA / "final.jsonl").exists(): prepare()
    train()
    gate_path = RUNS / "curriculum_calibration_gate.json"
    gate = load_bound_gate(gate_path, ADAPTERS / "curriculum", "calibration", "curriculum_calibration") if gate_path.exists() else evaluate_adapter(ADAPTERS / "curriculum", "calibration", "curriculum_calibration")
    selected = ADAPTERS / "curriculum"
    if not gate["pass"]:
        if corrective_allowed(gate):
            if not phase_complete(ADAPTERS / "curriculum", ADAPTERS / "corrective", CFG["phase_corrective"], "corrective", 55000): corrective()
            gate_path = RUNS / "corrective_calibration_gate.json"
            gate = load_bound_gate(gate_path, ADAPTERS / "corrective", "calibration", "corrective_calibration") if gate_path.exists() else evaluate_adapter(ADAPTERS / "corrective", "calibration", "corrective_calibration")
            selected = ADAPTERS / "corrective"
    if gate["pass"]:
        final_path = RUNS / "atomic_v5_final_gate.json"
        if final_path.exists():
            final_gate = load_bound_gate(final_path, selected, "final", "atomic_v5_final")
            finalize_final_access(selected, "atomic_v5_final", final_path)
        else:
            final_gate = evaluate_adapter(selected, "final", "atomic_v5_final")
        status = "PASS" if final_gate["pass"] else "FAILED_FINAL_GATE"
    else:
        final_gate = None; status = "FAILED_CALIBRATION_GATE"
    decision = {"schema": "stage4.sh1.v5.decision.v1", "status": status, "selected_adapter": str(selected) if gate["pass"] else None,
                "calibration_gate": gate, "final_gate": final_gate, "composition_unlocked": bool(final_gate and final_gate["pass"])}
    dump_json(RUNS / "ATOMIC_DECISION.json", decision); print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "seal-eval", "seal-training", "train", "calibrate", "corrective", "run")); command = parser.parse_args().command
    if command == "prepare": prepare()
    elif command == "seal-eval": print(json.dumps(seal_eval_exclusion(), indent=2))
    elif command == "seal-training": print(json.dumps(seal_training(), indent=2))
    elif command == "train": train()
    elif command == "calibrate": print(json.dumps(evaluate_adapter(ADAPTERS / "curriculum", "calibration", "curriculum_calibration"), indent=2))
    elif command == "corrective": corrective()
    else: run()


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
V1 = REPO / "artifacts/stage4_distill_v1/code"
V2 = REPO / "artifacts/stage4_distill_v2/code"
sys.path[:0] = [str(V2), str(V1)]

import v2_atomic as v2
from stage4_core import OPS, TOKENS, apply_prompt, dump_json, format_state, plan_prompt, read_jsonl, sha256, trajectory, write_jsonl

CFG = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
DATA = ROOT / "data"
RUNS = ROOT / "runs"
ADAPTERS = ROOT / "adapters"
MANIFESTS = ROOT / "manifests"
MODES = ("dense", "sparse", "monomial", "boundary", "alternating", "progression")


def state_key(p, state):
    return p, tuple(state)


def task_id(split, op, p, state):
    raw = json.dumps([split, op, p, list(state)], separators=(",", ":"))
    return "s4v3-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def make_state(rng, p, degree, mode):
    n = degree + 1
    if mode == "dense":
        state = [rng.randrange(p) for _ in range(n)]
    elif mode == "sparse":
        state = [0] * n
        for i in rng.sample(range(n), k=min(n, rng.choice((1, 2)))):
            state[i] = rng.randrange(1, p)
    elif mode == "monomial":
        state = [0] * n
        state[rng.randrange(n)] = rng.randrange(1, p)
    elif mode == "boundary":
        state = [rng.choice((0, 1, 2 % p, p - 1, max(0, p - 2))) for _ in range(n)]
    elif mode == "alternating":
        a = rng.randrange(1, p)
        state = [a if i % 2 == 0 else (-a) % p for i in range(n)]
    elif mode == "progression":
        a, d = rng.randrange(p), rng.randrange(1, p)
        state = [(a + i * d) % p for i in range(n)]
    else:
        raise ValueError(mode)
    return tuple(state)


def old_states():
    result = set()
    for path in (REPO / "artifacts/stage4_distill_v2/data").glob("*.jsonl"):
        result.update(state_key(r["p"], r["start"]) for r in read_jsonl(path))
    return result


def generate_rows(split, counts, seed, forbidden):
    rng = random.Random(seed)
    used = set(forbidden)
    rows = []
    for op in OPS:
        count = counts[op]
        degree_counts = ({1: 300, 2: 6500, 3: 8000, 4: 8500, 5: 8350, 6: 8350} if count == 40000
                         else ({1: 50, 2: 350, 3: 450, 4: 500, 5: 575, 6: 575} if count == 2500
                               else {degree: count // len(CFG["degrees"]) for degree in CFG["degrees"]}))
        if count == CFG["test_per_operation"]:
            degree_counts = {degree: count // len(CFG["degrees"]) for degree in CFG["degrees"]}
            for degree in CFG["degrees"][:count % len(CFG["degrees"])]:
                degree_counts[degree] += 1
        degree_schedule = [degree for degree in CFG["degrees"] for _ in range(degree_counts[degree])]
        assert len(degree_schedule) == count
        got = 0
        attempts = 0
        while got < count:
            attempts += 1
            if attempts > count * 10000:
                raise RuntimeError(f"generation exhausted for {split}/{op}: {got}/{count}")
            degree = degree_schedule[got]
            mode = rng.choice(MODES)
            p = CFG["fields"][rng.randrange(len(CFG["fields"]))]
            state = make_state(rng, p, degree, mode)
            key = state_key(p, state)
            if key in used or all(x == 0 for x in state):
                continue
            used.add(key)
            target = trajectory(state, [op], p)[-1]
            rows.append({"schema": "stage4.v3.atomic-task.v1", "task_id": task_id(split, op, p, state),
                         "split": split, "p": p, "degree": degree, "state_mode": mode,
                         "start": list(state), "operation": op, "program": [op], "witness": [op], "target": list(target)})
            got += 1
    rng.shuffle(rows)
    return rows, used


def prepare():
    for d in (DATA, RUNS, ADAPTERS, MANIFESTS):
        d.mkdir(parents=True, exist_ok=True)
    forbidden = old_states()
    train, used = generate_rows("targeted_train", CFG["train_counts"], 53001, forbidden)
    test_counts = {op: CFG["test_per_operation"] for op in OPS}
    test, used = generate_rows("fresh_atomic_test", test_counts, 53002, used)
    write_jsonl(DATA / "targeted_train.jsonl", train)
    write_jsonl(DATA / "fresh_atomic_test.jsonl", test)
    train_keys = {state_key(r["p"], r["start"]) for r in train}
    test_keys = {state_key(r["p"], r["start"]) for r in test}
    audit = {
        "status": "PASS", "old_state_overlap_train": len(forbidden & train_keys),
        "old_state_overlap_test": len(forbidden & test_keys), "train_test_state_overlap": len(train_keys & test_keys),
        "train_rows": len(train), "test_rows": len(test),
        "train_by_operation": Counter(r["operation"] for r in train),
        "test_by_operation": Counter(r["operation"] for r in test),
        "train_by_degree": Counter(r["degree"] for r in train),
        "train_by_mode": Counter(r["state_mode"] for r in train),
    }
    assert audit["old_state_overlap_train"] == audit["old_state_overlap_test"] == audit["train_test_state_overlap"] == 0
    assert audit["train_by_operation"] == Counter(CFG["train_counts"])
    dump_json(MANIFESTS / "data_audit.json", audit)
    manifest = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size, "rows": len(read_jsonl(p))} for p in DATA.glob("*.jsonl")}
    dump_json(MANIFESTS / "data_manifest.json", manifest)
    dump_json(MANIFESTS / "environment.json", {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
              "parent_adapter": CFG["parent_adapter"], "seed": CFG["seed"], "torch": v2.torch.__version__,
              "cuda": v2.torch.version.cuda, "gpu": v2.torch.cuda.get_device_name(0)})
    print(json.dumps(audit, indent=2, default=dict))


def triangle(degree):
    return "; ".join(f"i={i}:" + ",".join(str(math.comb(i, j)) for j in range(i + 1)) for i in range(degree + 1))


def scaffold_prompt(row):
    return ("SH1 RULE: b_j = sum over i>=j of c_i*binomial(i,j), reduced modulo FIELD. "
            f"BINOMIAL ROWS THROUGH DEGREE {row['degree']}: {triangle(row['degree'])}\n\n" + apply_prompt(row))


def encode_training(rows):
    rng = random.Random(53003)
    sh1 = [r for r in rows if r["operation"] == "SH1"]
    scaffold_ids = {r["task_id"] for r in rng.sample(sh1, CFG["sh1_scaffold"])}
    encoded = []
    for row in rows:
        prompt = scaffold_prompt(row) if row["task_id"] in scaffold_ids else apply_prompt(row)
        encoded.append(v2.encode_pair(v2.TOKENIZER, prompt, format_state(row["target"])))
    rng.shuffle(encoded)
    return encoded


def configure_v2_outputs():
    v2.RUNS = RUNS
    v2.DATA = DATA
    v2.ADAPTERS = ADAPTERS
    v2.MANIFESTS = MANIFESTS


def train():
    assert (DATA / "fresh_atomic_test.jsonl").exists(), "prepare and freeze test before training"
    configure_v2_outputs()
    model, _ = v2.load_continuation(REPO / CFG["parent_adapter"])
    spec = {"epochs": CFG["training"]["epochs"], "lr": CFG["training"]["lr"],
            "effective_batch": CFG["training"]["effective_batch"], "warmup_ratio": CFG["training"]["warmup_ratio"],
            "max_grad_norm": CFG["training"]["max_grad_norm"]}
    receipt = v2.train_phase(model, encode_training(read_jsonl(DATA / "targeted_train.jsonl")), spec, "targeted_pass")
    out = ADAPTERS / "atomic_targeted"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    v2.TOKENIZER.save_pretrained(out)
    dump_json(out / "training_receipt.json", {"status": "DONE", "parent": CFG["parent_adapter"], "phase": receipt})
    print(json.dumps(receipt, indent=2))


def evaluate():
    assert (ADAPTERS / "atomic_targeted/training_receipt.json").exists()
    configure_v2_outputs()
    test = read_jsonl(DATA / "fresh_atomic_test.jsonl")
    model, _ = v2.load_continuation(ADAPTERS / "atomic_targeted")
    gc.collect()
    if v2.torch.cuda.is_available(): v2.torch.cuda.empty_cache()
    model.float(); model.eval(); started = time.time()
    plan = v2.atomic_plan_accuracy(model, v2.TOKENIZER, v2.TIDS, test, batch_size=16)
    apply, generations = v2.atomic_apply_accuracy(model, v2.TOKENIZER, test, batch_size=8, return_generations=True)
    gate = v2.atomic_gate(plan, apply)
    gate.update({"run_id": "atomic_targeted_fresh_test", "evaluation_dtype": "float32", "wall_seconds": time.time() - started})
    write_jsonl(RUNS / "generations/atomic_targeted_fresh_test.jsonl", generations)
    dump_json(RUNS / "atomic_targeted_fresh_test_gate.json", gate)
    previous = json.loads((REPO / "artifacts/stage4_distill_v2/runs/atomic_hard_cycle_gate.json").read_text())
    decision = {"schema": "stage4.v3.atomic-decision.v1", "status": "PASS" if gate["pass"] else "FAILED",
                "composition_unlocked": bool(gate["pass"]), "fresh_test_gate": gate,
                "calibration_reference": {"sh1_apply": previous["apply"]["by_operation"]["SH1"], "overall_apply": previous["apply"]["overall"]}}
    dump_json(RUNS / "ATOMIC_DECISION.json", decision)
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "run"))
    command = parser.parse_args().command
    if command in ("prepare", "run") and not (DATA / "targeted_train.jsonl").exists(): prepare()
    if command in ("train", "run") and not (ADAPTERS / "atomic_targeted/training_receipt.json").exists(): train()
    if command in ("evaluate", "run"): evaluate()


if __name__ == "__main__":
    main()

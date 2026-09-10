from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

OPS = ("SH1", "SC2", "REV", "AC1", "AX1")
TOKENS = {op: f"<OP{i}>" for i, op in enumerate(OPS)}
INV_TOKENS = {v: k for k, v in TOKENS.items()}
HELDOUT = (("AX1", "SH1"), ("AC1", "REV"), ("SC2", "AX1"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict], *, gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    mode = "wt"
    with opener(path, mode, encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_op(state: Sequence[int], op: str, p: int) -> tuple[int, ...]:
    s = tuple(int(x) % p for x in state)
    if op == "SH1":
        from math import comb
        out = [0] * len(s)
        for i, ci in enumerate(s):
            for j in range(i + 1):
                out[j] = (out[j] + ci * comb(i, j)) % p
        return tuple(out)
    if op == "SC2":
        return tuple(ci * pow(2, i, p) % p for i, ci in enumerate(s))
    if op == "REV":
        return tuple(reversed(s))
    out = list(s)
    if op == "AC1":
        out[0] = (out[0] + 1) % p
    elif op == "AX1":
        out[1] = (out[1] + 1) % p
    else:
        raise ValueError(op)
    return tuple(out)


def trajectory(start: Sequence[int], program: Sequence[str], p: int) -> list[tuple[int, ...]]:
    states = [tuple(start)]
    for op in program:
        states.append(apply_op(states[-1], op, p))
    return states


def enumerate_programs(depth: int) -> list[tuple[str, ...]]:
    if depth not in (1, 2, 3, 4):
        raise ValueError("depth must be 1..4")
    return list(itertools.product(OPS, repeat=depth))


def motif_count(program: Sequence[str]) -> int:
    return sum(tuple(program[i:i+2]) in HELDOUT for i in range(len(program) - 1))


def format_state(state: Sequence[int]) -> str:
    return "[" + ", ".join(map(str, state)) + "]"


def plan_prompt(row: dict) -> str:
    return (f"FIELD: {row['p']}\nDEGREE_CAP: {len(row['start'])-1}\n"
            f"START: {format_state(row['start'])}\nTARGET: {format_state(row['target'])}\n"
            f"ALLOWED: {' '.join(TOKENS[o] for o in OPS)}\nMAX_STEPS: {row['depth']}\n"
            "Return exactly one line:\nPROGRAM:")


def apply_prompt(row: dict) -> str:
    return (f"FIELD: {row['p']}\nDEGREE_CAP: {len(row['start'])-1}\n"
            f"START: {format_state(row['start'])}\nPROGRAM: {' '.join(TOKENS[o] for o in row['program'])}\n"
            "Return exactly one line:\nRESULT:")


def program_answer(program: Sequence[str]) -> str:
    return " ".join(TOKENS[o] for o in program)


def trace_answer(states: Sequence[Sequence[int]]) -> str:
    return " TRACE: " + " -> ".join(format_state(s) for s in states[1:])


def _task_id(split: str, p: int, start: Sequence[int], program: Sequence[str], nonce: int) -> str:
    raw = json.dumps([split, p, list(start), list(program), nonce], separators=(",", ":"))
    return "s4-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def generate_split(name: str, n: int, fields: Sequence[int], depths: Sequence[int], seed: int,
                   motif_rule: str = "none", forbidden_states: set | None = None) -> list[dict]:
    rng = random.Random(seed)
    forbidden_states = forbidden_states if forbidden_states is not None else set()
    rows, used_ids, used_endpoints = [], set(), set()
    attempts = 0
    while len(rows) < n:
        attempts += 1
        if attempts > n * 500:
            raise RuntimeError(f"could not generate {name}: {len(rows)}/{n}")
        p = rng.choice(tuple(fields)); depth = rng.choice(tuple(depths)); degree = rng.choice((2, 3, 4))
        start = tuple(rng.randrange(p) for _ in range(degree + 1))
        if all(x == 0 for x in start):
            continue
        program = tuple(rng.choice(OPS) for _ in range(depth))
        mc = motif_count(program)
        if motif_rule == "none" and mc != 0:
            continue
        if motif_rule == "exactly_one" and mc != 1:
            continue
        states = trajectory(start, program, p)
        endpoint_key = (p, start, states[-1], depth)
        state_keys = {(p, tuple(s)) for s in states}
        if endpoint_key in used_endpoints or state_keys & forbidden_states:
            continue
        tid = _task_id(name, p, start, program, attempts)
        if tid in used_ids:
            continue
        solutions = []
        for candidate_depth in range(1, depth + 1):
            at_depth=[]
            for candidate in enumerate_programs(candidate_depth):
                if trajectory(start, candidate, p)[-1] == states[-1]:
                    at_depth.append(list(candidate))
                    if len(at_depth) == 2: break
            if at_depth:
                solutions=at_depth
                break
        row = {"schema":"stage4.task.v1","task_id":tid,"split":name,"p":p,"depth":depth,
               "start":list(start),"target":list(states[-1]),"witness":list(program),
               "solutions":solutions,"states":[list(s) for s in states],"motif_count":mc}
        rows.append(row); used_ids.add(tid); used_endpoints.add(endpoint_key); forbidden_states.update(state_keys)
    return rows


def split_audit(splits: dict[str, list[dict]]) -> dict:
    ids, states, leaks = {}, {}, []
    for name, rows in splits.items():
        ids[name] = {r["task_id"] for r in rows}
        states[name] = {(r["p"], tuple(s)) for r in rows for s in r["states"]}
        if name == "train":
            leaks.extend(r["task_id"] for r in rows if motif_count(r["witness"]) != 0)
    id_overlap, state_overlap = {}, {}
    for a, b in itertools.combinations(splits, 2):
        id_overlap[f"{a}:{b}"] = len(ids[a] & ids[b])
        state_overlap[f"{a}:{b}"] = len(states[a] & states[b])
    return {"task_id_overlap":id_overlap,"state_overlap":state_overlap,
            "train_heldout_motif_leaks":leaks,"leakage_count":len(leaks),
            "ok":not leaks and not any(id_overlap.values()) and not any(state_overlap.values())}


def discover(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    seen, out = set(), []
    pos = Counter(); first = Counter(); motifs = Counter(); sigs = Counter()
    for row in rows:
        sig = tuple(row["witness"])
        key = (row["p"], tuple(row["start"]), sig, tuple(row["target"]))
        if key in seen:
            continue
        seen.add(key)
        states = trajectory(row["start"], sig, row["p"])
        rec = {"schema":"stage4.trajectory.v1","task_id":row["task_id"],"p":row["p"],
               "start":row["start"],"target":row["target"],"program":list(sig),
               "states":[list(s) for s in states]}
        out.append(rec); sigs[" ".join(sig)] += 1; first[sig[0]] += 1
        for i, op in enumerate(sig): pos[f"{i}:{op}"] += 1
        for i in range(len(sig)-1): motifs[f"{sig[i]}>{sig[i+1]}"] += 1
    pools=defaultdict(list)
    for r in out: pools[(r["program"][0],len(r["program"]))].append(r)
    for pool in pools.values(): pool.sort(key=lambda x:hashlib.sha256(x["task_id"].encode()).hexdigest())
    quota=min(150,min(map(len,pools.values())))
    # Greedy stratification gives exact first-op and depth balance, then selects the candidate
    # that currently uses the least represented local motifs (deterministic hash tie-break).
    selected=[]; balance_motifs=Counter(); balance_signatures=Counter()
    for _ in range(quota):
        for key in sorted(pools):
            pool=pools[key]
            def balance_score(r):
                sig=r["program"]; ms=[f"{sig[i]}>{sig[i+1]}" for i in range(len(sig)-1)]
                return (sum(balance_motifs[m] for m in ms)/len(ms),balance_signatures[" ".join(sig)],hashlib.sha256(r["task_id"].encode()).hexdigest())
            best=min(range(len(pool)),key=lambda i:balance_score(pool[i])); r=pool.pop(best); selected.append(r)
            sig=r["program"]; balance_signatures[" ".join(sig)]+=1
            for i in range(len(sig)-1): balance_motifs[f"{sig[i]}>{sig[i+1]}"]+=1
    out=selected
    first=Counter(); pos=Counter(); motifs=Counter(); sigs=Counter()
    for r in out:
        sig=tuple(r["program"]); first[sig[0]]+=1; sigs[" ".join(sig)]+=1
        for i,op in enumerate(sig): pos[f"{i}:{op}"]+=1
        for i in range(len(sig)-1): motifs[f"{sig[i]}>{sig[i+1]}"]+=1
    motif_values=list(motifs.values())
    required_positions=[f"{i}:{op}" for i in range(4) for op in OPS]
    min_position=min(pos.get(k,0) for k in required_positions)
    report = {"unique_trajectories":len(out),"solvable_tasks":len({r['task_id'] for r in out}),
              "full_signatures":len(sigs),"first_operation":first,"position_operation":pos,
              "local_motifs":motifs,"balance":{"method":"exact first-op/depth strata plus deterministic least-represented-motif greedy selection",
              "first_op_spread":max(first.values())-min(first.values()),"local_motif_max_min_ratio":max(motif_values)/min(motif_values),
              "minimum_operation_position_count":min_position},
              "passes_minimums":len(out)>=2000 and len({r['task_id'] for r in out})>=1000 and len(sigs)>=20 and min_position>=50}
    return out, report


@dataclass
class RankingMetrics:
    hit: dict[str, float]
    correct_mass: float
    best_rank: int
    mrr: float
    log_gap: float


def metrics_from_scores(programs: Sequence[Sequence[str]], scores: Sequence[float], correct: Sequence[bool]) -> RankingMetrics:
    ranked = sorted(zip(programs, scores, correct), key=lambda x: (-x[1], tuple(x[0])))
    correct_ranks = [i + 1 for i, x in enumerate(ranked) if x[2]]
    if not correct_ranks:
        return RankingMetrics({f"hit@{k}":0.0 for k in (1,8,16,32,64)},0.0,0,0.0,float("-inf"))
    mx = max(scores); z = sum(math.exp(s-mx) for s in scores)
    cm = sum(math.exp(s-mx) for s,c in zip(scores,correct) if c) / z
    best_good = max(s for s,c in zip(scores,correct) if c)
    bad = [s for s,c in zip(scores,correct) if not c]
    gap = best_good - max(bad) if bad else float("inf")
    br = min(correct_ranks)
    return RankingMetrics({f"hit@{k}":float(br<=k) for k in (1,8,16,32,64)},cm,br,1.0/br,gap)

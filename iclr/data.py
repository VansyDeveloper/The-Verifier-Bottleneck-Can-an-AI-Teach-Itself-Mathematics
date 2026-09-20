"""Fresh Stage4 tasks and fixed-exposure supervised mixtures."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import composition_core as core


def constraint_hits(program, row):
    pairs = {tuple(p) for p in row.get('heldout_motifs', core.HELDOUT_MOTIFS)}
    triples = {tuple(p) for p in row.get('heldout_triples', [])}
    positions = {tuple(p) for p in row.get('heldout_positions', [])}
    return (sum(tuple(p) in pairs for p in zip(program, program[1:]))
            + sum(tuple(p) in triples for p in zip(program, program[1:], program[2:]))
            + sum((i, a, b) in positions for i, (a, b) in enumerate(zip(program, program[1:]))))


def correct_programs(row):
    """Return the full correct set at the declared depth, never the two witnesses."""
    programs = [program for program in core.enumerate_programs(int(row["depth"]))
                if core.verify_program(row["start"], row["target"], program, int(row["p"]))]
    if not programs:
        raise ValueError(f"task has no correct program: {row.get('task_id')}")
    if str(row.get("family", "")).upper() == "TRAIN" and any(constraint_hits(p, row) for p in programs):
        raise ValueError(f"held motif in full train correct set: {row.get('task_id')}")
    return programs


def assign_witnesses(examples, policy, seed):
    """Change only correct PROGRAM labels; task identities and exposure order stay fixed."""
    if policy not in ('fixed', 'uniform', 'balanced'):
        raise ValueError(f'Unknown witness policy: {policy}')
    if policy == 'fixed':
        return examples
    rng, counts, cache = random.Random(seed), Counter(), {}
    result = []
    for item in examples:
        if item['kind'] != 'composition':
            raise ValueError('Witness intervention requires PROGRAM-only composition tasks')
        row = item['row']
        if row['task_id'] not in cache:
            cache[row['task_id']] = correct_programs(row)
        candidates = list(cache[row['task_id']])
        rng.shuffle(candidates)
        if policy == 'uniform':
            program = candidates[0]
        else:
            # ponytail: greedy reduction of pair-count imbalance, not a global optimum.
            # Exact integer optimization is only needed if this intervention has no effect.
            def cost(program):
                added = Counter(zip(program, program[1:]))
                return sum(2 * counts[pair] * n + n * n for pair, n in added.items())
            program = min(candidates, key=cost)
        counts.update(zip(program, program[1:]))
        selected = {**row, 'witness': list(program), 'program': list(program),
                    'states': [list(s) for s in core.trajectory(row['start'], program, row['p'])]}
        result.append({**item, 'row': selected, 'answer': render_target(selected, 'program_only')})
    return result


def render_target(row, target_format):
    if target_format not in ("program_only", "program_trace"):
        raise ValueError(f"unknown target format: {target_format}")
    program = row.get("program") or row["witness"]
    states = core.trajectory(row["start"], program, int(row["p"]))
    if len(program) != int(row["depth"]) or list(states[-1]) != row["target"]:
        raise ValueError(f"invalid witness: {row.get('task_id')}")
    if "states" in row and [list(s) for s in states] != row["states"]:
        raise ValueError(f"invalid stored states: {row.get('task_id')}")
    answer = core.program_answer(program)
    if target_format == "program_trace":
        answer += "\nTRACE: " + " -> ".join(core.format_state(s) for s in states[1:])
    return answer


def _stratified_order(rows, seed):
    pools = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pools[int(row["depth"])][int(row["p"])].append(row)
    rng = random.Random(seed)
    ordered = []
    for depth, fields in sorted(pools.items()):
        within_depth = []
        for field, pool in sorted(fields.items()):
            pool.sort(key=lambda row: row["task_id"])
            rng.shuffle(pool)
            within_depth.extend(((i + 0.5) / len(pool), rng.random(), row) for i, row in enumerate(pool))
        within_depth.sort(key=lambda entry: entry[:2])
        # Nested prefixes retain depth ratios, then field proportions within depth.
        ordered.extend(((i + 0.5) / len(within_depth), rng.random(), entry[2])
                       for i, entry in enumerate(within_depth))
    return [entry[2] for entry in sorted(ordered, key=lambda entry: entry[:2])]


def mixture(compositions, atomic_rows, size, replay, seed, target_format):
    """Replace composition exposure slots; balance atomic PLAN/APPLY x operation."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    if not math.isfinite(replay) or not 0 <= replay <= 1:
        raise ValueError("replay must be between 0 and 1")
    if target_format not in ("program_only", "program_trace"):
        raise ValueError(f"unknown target format: {target_format}")
    atomic_n = round(size * replay)
    composition_n = size - atomic_n
    if composition_n and not compositions:
        raise ValueError("composition training needs a nonempty task pool")
    if composition_n and any(row.get("family") != "TRAIN" for row in compositions):
        raise ValueError("composition training requires TRAIN rows")
    if len({row["task_id"] for row in compositions}) != len(compositions):
        raise ValueError("duplicate composition task IDs")
    ordered = _stratified_order(compositions, seed)
    selected = [ordered[index % len(ordered)] for index in range(composition_n)]
    specs = [{"prompt": core.plan_prompt(row), "answer": render_target(row, target_format),
              "kind": "composition", "operation": None, "task_id": row["task_id"], "row": row}
             for row in selected]
    if atomic_n:
        pools = defaultdict(list)
        for row in atomic_rows:
            op = row.get("operation")
            if op not in core.OPS or row.get("family") != "ATOMIC" or row.get("depth") != 1:
                raise ValueError("atomic training requires depth-one ATOMIC rows with an operation")
            if (row.get("program") or row.get("witness")) != [op]:
                raise ValueError("atomic operation and witness disagree")
            render_target(row, "program_only")
            pools[op].append(row)
        if set(pools) != set(core.OPS):
            raise ValueError("atomic training needs nonempty pools for all five operations")
        if len({row["task_id"] for row in atomic_rows}) != len(atomic_rows):
            raise ValueError("duplicate atomic task IDs")
        rng = random.Random(seed)
        for pool in pools.values():
            pool.sort(key=lambda row: row["task_id"])
            rng.shuffle(pool)
        cells = [(op, kind) for op in core.OPS for kind in ("atomic_plan", "atomic_apply")]
        rng.shuffle(cells)
        for index in range(atomic_n):
            op, kind = cells[index % len(cells)]
            pool = pools[op]
            row = pool[(index // len(cells)) % len(pool)]
            prompt = core.plan_prompt(row) if kind == "atomic_plan" else core.apply_prompt(row)
            answer = core.program_answer([op]) if kind == "atomic_plan" else core.format_state(row["target"])
            specs.append({"prompt": prompt, "answer": answer, "kind": kind, "operation": op,
                          "task_id": row["task_id"], "row": row})
    random.Random(seed).shuffle(specs)
    for index, spec in enumerate(specs):
        spec["exposure_index"] = index
    return specs


def _full_states(row, programs):
    return {core.state_fingerprint(row["p"], state) for program in programs
            for state in core.trajectory(row["start"], program, row["p"])}


def _composition_split(name, count, family, seed, registry, depth_counts=None):
    schedule = [depth for depth, n in (depth_counts or {3: count}).items() for _ in range(n)]
    rng = random.Random(seed)
    rng.shuffle(schedule)
    rows = []
    for depth in schedule:
        for _ in range(10000):
            row = core.generate_split(name, 1, family, seed=rng.randrange(2**63),
                                      depths=(depth,), registry=registry)[0]
            # The frozen generator reserves two witnesses. Replace that reservation
            # with every correct trajectory before accepting the next task.
            registry.states.difference_update(core._row_state_fingerprints(row))
            registry.tasks.remove(row["task_fingerprint"])
            programs = correct_programs(row)
            states = _full_states(row, programs)
            if states & registry.states:
                continue
            registry.states.update(states)
            registry.tasks.add(row["task_fingerprint"])
            row["degree"] = len(row["start"]) - 1
            row["correct_count"] = len(programs)
            rows.append(row)
            break
        else:
            raise RuntimeError(f"full-correct-set state exclusions exhausted for {name}")
    return rows


def _atomic_split(name, per_operation, seed, registry):
    rng = random.Random(seed)
    rows = []
    for op in core.OPS:
        for _ in range(per_operation):
            for _ in range(20000):
                p, degree = rng.choice(core.KNOWN_FIELDS), rng.choice(core.DEGREES)
                start = [rng.randrange(p) for _ in range(degree + 1)]
                states = core.trajectory(start, [op], p)
                if not any(start) or states[0] == states[-1]:
                    continue
                fingerprints = {core.state_fingerprint(p, state) for state in states}
                if fingerprints & registry.states:
                    continue
                target = list(states[-1])
                fingerprint = core.canonical_task_fingerprint(p, start, target, 1)
                if fingerprint in registry.tasks:
                    continue
                search = core.exact_shortest_solutions(start, target, p, 1)
                if search.total_solutions != 1:
                    continue
                row = {"schema": core.TASK_SCHEMA, "task_id": "s4c-" + fingerprint[:24],
                       "task_fingerprint": fingerprint, "split": name, "family": "ATOMIC",
                       "p": p, "degree": degree, "depth": 1, "operation": op,
                       "start": start, "target": target, "witness": [op],
                       "states": [list(state) for state in states], "motif_count": 0,
                       "shortest_depth": 1, "shortest_solution_count": search.total_solutions,
                       "correct_count": search.total_solutions,
                       "solutions": [solution.as_record() for solution in search.solutions]}
                registry.states.update(fingerprints)
                registry.tasks.add(fingerprint)
                rows.append(row)
                break
            else:
                raise RuntimeError(f"atomic generation exhausted for {name}, {op}")
    rng.shuffle(rows)
    return rows


def generate(out, size=5000, eval_size=200, atomic_per_op=1000, seed=20260910, smoke=False):
    out = Path(out)
    if min(size, eval_size, atomic_per_op) <= 0 or size % 4:
        raise ValueError("positive sizes required; train size must be divisible by four")
    if not smoke and size < 5000:
        raise ValueError("full runs need at least 5000 composition tasks; use --smoke for checks")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty data directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    registry = core.FingerprintRegistry()
    files = {}

    def save(name, rows):
        payload = core.canonical_jsonl_bytes(rows)
        (out / f"{name}.jsonl").write_bytes(payload)
        files[f"{name}.jsonl"] = {"rows": len(rows), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "field_depth_counts": dict(sorted(Counter(f"{r['p']}:{r['depth']}" for r in rows).items())),
            "operation_counts": dict(sorted(Counter(r["operation"] for r in rows if "operation" in r).items()))}

    save("train", _composition_split("train", size, "TRAIN", seed, registry,
                                     {2: size // 4, 3: size // 2, 4: size // 4}))
    save("atomic_train", _atomic_split("atomic_train", atomic_per_op, seed + 1, registry))
    for phase_index, phase in enumerate(("dev", "final")):
        for family_index, family in enumerate("ABCD"):
            name = f"{phase}_{family}"
            save(name, _composition_split(name, eval_size, family,
                                          seed + 10 + phase_index * 10 + family_index, registry))
        name = f"{phase}_atomic"
        save(name, _atomic_split(name, eval_size, seed + 14 + phase_index * 10, registry))
    manifest = {"schema": "iclr.data.v1", "status": "smoke" if smoke else "fresh_screening_data",
                "historical_reproduction": False,
                "config": {"size": size, "eval_size": eval_size, "atomic_per_op": atomic_per_op,
                           "seed": seed, "smoke": smoke},
                "state_exclusion": "all states of all correct programs, across all tasks and splits",
                "heldout_motifs": core.HELDOUT_MOTIFS,
                "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=200, help="tasks per A-D split; tasks per atomic operation")
    parser.add_argument("--atomic-per-op", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--smoke", action="store_true", help="use 20 train, 2 eval and 5 atomic-train tasks per operation")
    args = parser.parse_args()
    if args.smoke:
        args.size, args.eval_size, args.atomic_per_op = 20, 2, 5
    manifest = generate(**vars(args))
    print(json.dumps({"out": str(args.out), "status": manifest["status"],
                      "rows": {name: entry["rows"] for name, entry in manifest["files"].items()}}))


if __name__ == "__main__":
    main()

"""Preselect feasible pair-withholding sets for a future experiment only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/data/discover_trajectories.jsonl"


def build_design(rows: list[dict], *, ks: tuple[int, ...] = (1, 2, 3, 5),
                 subsets_per_k: int = 5, minimum_per_depth: int = 250,
                 seed: int = 84000) -> dict:
    if not rows or subsets_per_k <= 0 or minimum_per_depth <= 0:
        raise ValueError("invalid input")
    universe = sorted({tuple(pair) for row in rows
                       for pair in zip(row["program"], row["program"][1:])})
    rng = random.Random(seed)
    groups = []
    for k in ks:
        if k <= 0 or k > len(universe):
            raise ValueError("invalid withheld-pair count")
        chosen = set()
        entries = []
        for _ in range(100000):
            if len(entries) == subsets_per_k:
                break
            pairs = tuple(sorted(rng.sample(universe, k)))
            if pairs in chosen:
                continue
            chosen.add(pairs)
            remaining = Counter(row["depth"] for row in rows
                                if not any(pair in pairs for pair in
                                           zip(row["program"], row["program"][1:])))
            if any(remaining[depth] < minimum_per_depth for depth in (2, 3, 4)):
                continue
            entries.append({"subset": len(entries) + 1,
                            "pairs": [list(pair) for pair in pairs],
                            "remaining_by_depth": {str(depth): remaining[depth]
                                                   for depth in (2, 3, 4)}})
        if len(entries) != subsets_per_k:
            raise ValueError(f"could not find {subsets_per_k} feasible sets for k={k}")
        groups.append({"k": k, "sets": entries})
    return {"schema": "verifier-bottleneck.future-withheld-pairs.v1",
            "status": "DESIGN_ONLY_NO_TRAINING", "draw_seed": seed,
            "pair_universe": [list(pair) for pair in universe],
            "minimum_remaining_per_depth": minimum_per_depth,
            "k_groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "WITHHELD_PAIR_SETS.json")
    args = parser.parse_args()
    payload = SOURCE.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if len(rows) != 2250:
        raise RuntimeError("source pool is not the expected 2250 trajectories")
    design = build_design(rows)
    design["source_sha256"] = hashlib.sha256(payload).hexdigest().upper()
    content = json.dumps(design, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != content:
            raise RuntimeError("preselected set manifest already exists and differs")
    else:
        args.output.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": design["status"], "sets": sum(len(group["sets"])
                      for group in design["k_groups"]), "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

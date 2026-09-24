"""Matched data selection for the preselected withheld-pair study."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


def _allocate(capacities: dict[tuple, int], total: int) -> dict[tuple, int]:
    available = sum(capacities.values())
    if available < total:
        raise ValueError(f"insufficient matched pool: {available} < {total}")
    exact = {key: total * count / available for key, count in capacities.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    extra = total - sum(quotas.values())
    for key in sorted(capacities, key=lambda item: (-(exact[item] - quotas[item]), item))[:extra]:
        quotas[key] += 1
    return quotas


def build_selection(rows: list[dict], target_lengths: dict[str, int],
                    excluded_pairs: set[tuple[str, str]], *, seed: int,
                    per_depth: int = 250) -> dict:
    if per_depth <= 0 or not excluded_pairs:
        raise ValueError("invalid selection request")
    by_id = {row["trajectory_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(target_lengths):
        raise ValueError("duplicate trajectory or target-length mismatch")
    if any(not isinstance(value, int) or value <= 0 for value in target_lengths.values()):
        raise ValueError("invalid target length")
    forbidden = {identity for identity, row in by_id.items()
                 if any(pair in excluded_pairs for pair in zip(row["program"], row["program"][1:]))}
    grouped = defaultdict(list)
    for identity, row in by_id.items():
        grouped[(int(row["depth"]), int(row["p"]))].append(identity)
    rng = random.Random(seed)
    random_removed = set()
    removed_counts = Counter((int(by_id[identity]["depth"]), int(by_id[identity]["p"]))
                             for identity in forbidden)
    for key in sorted(grouped):
        count = removed_counts[key]
        random_removed.update(rng.sample(sorted(grouped[key]), count))

    pools = {}
    for arm, removed in (("withheld", forbidden), ("random", random_removed)):
        strata = defaultdict(list)
        for identity, row in by_id.items():
            if identity not in removed:
                strata[(int(row["depth"]), int(row["p"]), target_lengths[identity])].append(identity)
        pools[arm] = strata

    quotas = {}
    for depth in (2, 3, 4):
        keys = sorted(key for key in set(pools["withheld"]) | set(pools["random"]) if key[0] == depth)
        capacities = {key: min(len(pools["withheld"][key]), len(pools["random"][key])) for key in keys}
        quotas.update(_allocate(capacities, per_depth))

    selections = {}
    for arm in ("withheld", "random"):
        chosen = []
        for key in sorted(quotas):
            chosen.extend(rng.sample(sorted(pools[arm][key]), quotas[key]))
        selections[arm] = chosen
    left = Counter((int(by_id[identity]["depth"]), int(by_id[identity]["p"]), target_lengths[identity])
                   for identity in selections["withheld"])
    right = Counter((int(by_id[identity]["depth"]), int(by_id[identity]["p"]), target_lengths[identity])
                    for identity in selections["random"])
    if left != right:
        raise AssertionError("matched strata differ")
    return {
        "withheld_ids": selections["withheld"], "random_ids": selections["random"],
        "withheld_removed_ids": sorted(forbidden), "random_removed_ids": sorted(random_removed),
        "removed_by_depth_p": {f"{depth}:{p}": count for (depth, p), count in sorted(removed_counts.items())},
        "budget": {"records_each": len(selections["withheld"]),
                   "target_tokens_per_epoch_each": sum(target_lengths[identity] for identity in selections["withheld"]),
                   "strata": len(left)},
        "strata_quotas": [{"depth": key[0], "p": key[1], "target_tokens": key[2], "count": value}
                          for key, value in sorted(quotas.items()) if value],
    }


def build_all_selections(rows, target_lengths, design, *, seeds=(85000, 85001, 85002), per_depth=250):
    result = []
    for group in design["k_groups"]:
        for subset in group["sets"]:
            entry = {"k": group["k"], "subset": subset["subset"], "pairs": subset["pairs"]}
            excluded = {tuple(pair) for pair in subset["pairs"]}
            selections = {}
            try:
                for seed in seeds:
                    selections[str(seed)] = build_selection(rows, target_lengths, excluded,
                                                              seed=seed, per_depth=per_depth)
            except ValueError as exc:
                entry.update({"status": "INFEASIBLE", "reason": str(exc)})
            else:
                entry.update({"status": "FEASIBLE", "selections": selections})
            result.append(entry)
    return result

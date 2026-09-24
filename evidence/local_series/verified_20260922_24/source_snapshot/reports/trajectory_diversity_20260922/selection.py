"""Matched-budget random and coverage-oriented trajectory selection."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict


def state_class(p: int, values: list[int]) -> tuple[int, ...]:
    if len(values) not in (3, 4, 5) or p <= 0 or any(not 0 <= value < p for value in values):
        raise ValueError("invalid intermediate state")
    return (p, *(min(3, 4 * value // p) for value in values))


def _features(row: dict) -> tuple[tuple, tuple]:
    pairs = tuple(zip(row["program"], row["program"][1:]))
    states = tuple(state_class(int(row["p"]), state) for state in row["states"][1:-1])
    return pairs, states


def _tie(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest(), "big")


def _quotas(grouped: dict[tuple, list[dict]], total: int) -> dict[tuple, int]:
    if total > sum(map(len, grouped.values())):
        raise ValueError("requested more rows than available")
    size = sum(map(len, grouped.values()))
    exact = {key: total * len(values) / size for key, values in grouped.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(grouped, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def validate_pair_budget(rows: list[dict], target_lengths: dict[str, int],
                         random_ids: list[str], diverse_ids: list[str]) -> dict:
    by_id = {row["trajectory_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate source trajectory")
    for ids in (random_ids, diverse_ids):
        if len(ids) != len(set(ids)) or any(identity not in by_id for identity in ids):
            raise ValueError("selection contains duplicate or unknown trajectory")
    if len(random_ids) != len(diverse_ids):
        raise ValueError("different training record counts")
    def strata(ids):
        return Counter((by_id[identity]["depth"], by_id[identity]["p"], target_lengths[identity])
                       for identity in ids)
    left, right = strata(random_ids), strata(diverse_ids)
    if left != right:
        raise ValueError("different depth, modulus, or target-token strata")
    return {"records_each": len(random_ids),
            "target_tokens_per_epoch_each": sum(target_lengths[identity] for identity in random_ids),
            "strata": len(left)}


def select_pair(rows: list[dict], target_lengths: dict[str, int], *, seed: int,
                per_depth: int = 250, depths: tuple[int, ...] = (2, 3, 4)) -> dict:
    if per_depth <= 0 or not depths:
        raise ValueError("selection size must be positive")
    if len({row["trajectory_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate trajectory id")
    if set(target_lengths) != {row["trajectory_id"] for row in rows}:
        raise ValueError("target length map does not match trajectory pool")
    grouped = defaultdict(list)
    for row in rows:
        if row["depth"] in depths:
            grouped[(row["depth"], row["p"], target_lengths[row["trajectory_id"]])].append(row)
    quotas = {}
    for depth in depths:
        subset = {key: values for key, values in grouped.items() if key[0] == depth}
        if sum(map(len, subset.values())) < per_depth:
            raise ValueError(f"insufficient depth {depth} pool")
        quotas.update(_quotas(subset, per_depth))
    rng = random.Random(seed)
    random_ids = []
    for key in sorted(quotas):
        pool = sorted(grouped[key], key=lambda row: row["trajectory_id"])
        random_ids.extend(row["trajectory_id"] for row in rng.sample(pool, quotas[key]))
    features = {row["trajectory_id"]: _features(row) for row in rows}
    universe_pairs = {item for values in features.values() for item in values[0]}
    universe_states = {item for values in features.values() for item in values[1]}
    total_pairs = sum((depth - 1) * per_depth for depth in depths)
    pair_norm = len(universe_pairs) * math.log1p(total_pairs / len(universe_pairs))
    state_norm = len(universe_states)
    candidates = {key: sorted(grouped[key], key=lambda row: row["trajectory_id"]) for key in quotas}
    remaining = dict(quotas)
    pair_count = Counter()
    seen_states = set()
    diverse_ids = []
    for _ in range(per_depth * len(depths)):
        best = None
        best_gain = -math.inf
        best_tie = None
        best_key = None
        for key, pool in candidates.items():
            if remaining[key] == 0:
                continue
            for row in pool:
                identity = row["trajectory_id"]
                pairs, states = features[identity]
                local = Counter(pairs)
                pair_gain = sum(math.log1p(pair_count[pair] + count) - math.log1p(pair_count[pair])
                                for pair, count in local.items()) / pair_norm
                state_gain = len(set(states) - seen_states) / state_norm
                gain = 0.5 * pair_gain + 0.5 * state_gain
                tie = _tie(seed, identity)
                if gain > best_gain + 1e-15 or (abs(gain - best_gain) <= 1e-15 and (best_tie is None or tie < best_tie)):
                    best, best_gain, best_tie, best_key = row, gain, tie, key
        if best is None:
            raise ValueError("selection exhausted before filling quotas")
        identity = best["trajectory_id"]
        diverse_ids.append(identity)
        remaining[best_key] -= 1
        pair_count.update(features[identity][0])
        seen_states.update(features[identity][1])
        candidates[best_key] = [row for row in candidates[best_key] if row["trajectory_id"] != identity]
    def summary(ids: list[str]) -> dict:
        pairs = Counter(item for identity in ids for item in features[identity][0])
        states = {item for identity in ids for item in features[identity][1]}
        return {"pair_types": len(pairs), "minimum_pair_count": min(pairs.values()),
                "pair_balance_log_sum": sum(math.log1p(value) for value in pairs.values()) / pair_norm,
                "state_classes": len(states), "state_coverage_fraction": len(states) / state_norm,
                "selection_objective": 0.5 * sum(math.log1p(value) for value in pairs.values()) / pair_norm + 0.5 * len(states) / state_norm,
                "target_tokens_per_epoch": sum(target_lengths[identity] for identity in ids)}
    return {"random_ids": random_ids, "diverse_ids": diverse_ids,
            "random": summary(random_ids), "diverse": summary(diverse_ids),
            "strata_quotas": [{"depth": key[0], "p": key[1], "target_tokens": key[2], "count": value}
                              for key, value in sorted(quotas.items())]}

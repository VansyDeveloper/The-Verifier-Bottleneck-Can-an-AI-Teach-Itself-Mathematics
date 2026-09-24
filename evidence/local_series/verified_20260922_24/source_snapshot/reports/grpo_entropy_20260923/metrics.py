"""Predeclared group and exact-125-program metrics for the entropy ablation."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import torch


OPS = ("SH1", "SC2", "REV", "AC1", "AX1")


def normalized_entropy(scaled_scores: torch.Tensor) -> torch.Tensor:
    if scaled_scores.numel() != len(OPS):
        raise ValueError("expected five operation scores")
    log_prob = torch.log_softmax(scaled_scores.float(), dim=0)
    return -(log_prob.exp() * log_prob).sum() / math.log(len(OPS))


def group_summary(programs: list[tuple[str, ...]], rewards: list[bool]) -> dict:
    if len(programs) != 8 or len(rewards) != 8 or any(type(value) is not bool for value in rewards):
        raise ValueError("one group requires exactly eight boolean outcomes")
    counts = Counter(programs)
    correct = Counter(program for program, reward in zip(programs, rewards) if reward)
    correct_count = sum(rewards)
    entropy = -sum((n / 8) * math.log(n / 8) for n in counts.values())
    return {
        "group_class": "all_wrong" if correct_count == 0 else "all_correct" if correct_count == 8 else "mixed",
        "correct_count": correct_count,
        "unique_programs": len(counts),
        "unique_correct_programs": len(correct),
        "sample_entropy_nats": entropy,
        "collision_fraction": sum(n * (n - 1) // 2 for n in counts.values()) / 28,
    }


def all_programs() -> list[tuple[str, ...]]:
    return list(itertools.product(OPS, repeat=3))


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def exact_ranking_metrics(rows: list[tuple[tuple[str, ...], float, bool]]) -> tuple[dict, list[dict]]:
    programs = all_programs()
    if len(rows) != len(programs):
        raise ValueError("ranking requires exactly 125 programs")
    seen = set()
    normalized = []
    for program, raw_score, raw_correct in rows:
        program = tuple(program)
        if program in seen or program not in programs:
            raise ValueError("duplicate or invalid program")
        seen.add(program)
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score):
            raise ValueError("score must be finite")
        if type(raw_correct) is not bool:
            raise ValueError("correctness flag must be boolean")
        normalized.append({"program": list(program), "score": float(raw_score), "correct": raw_correct})
    if seen != set(programs):
        raise ValueError("ranking has missing programs")
    normalized.sort(key=lambda item: (-item["score"], tuple(item["program"])))
    for rank, item in enumerate(normalized, 1):
        item["rank"] = rank
    correct_rows = [item for item in normalized if item["correct"]]
    if not correct_rows:
        raise ValueError("task has no correct program")
    scores = [item["score"] for item in normalized]
    log_z = _logsumexp(scores)
    probabilities = [math.exp(score - log_z) for score in scores]
    correct_scores = [item["score"] for item in correct_rows]
    log_correct_z = _logsumexp(correct_scores)
    correct_probabilities = [math.exp(score - log_correct_z) for score in correct_scores]
    entropy = -sum(prob * math.log(prob) for prob in probabilities if prob)
    correct_entropy = -sum(prob * math.log(prob) for prob in correct_probabilities if prob)
    best_rank = min(item["rank"] for item in correct_rows)
    summary = {
        "best_rank": best_rank,
        "hit@32": float(best_rank <= 32),
        "correct_mass": math.exp(log_correct_z - log_z),
        "entropy_nats": entropy,
        "effective_programs": math.exp(entropy),
        "correct_entropy_nats": correct_entropy,
        "unique_correct_programs": len(correct_rows),
    }
    return summary, normalized


def validate_pair_config(control: dict, entropy: dict) -> None:
    if control.get("entropy_coef") != 0.0 or entropy.get("entropy_coef") != 0.01:
        raise ValueError("entropy coefficients must be 0 and 0.01")
    left = {key: value for key, value in control.items() if key != "entropy_coef"}
    right = {key: value for key, value in entropy.items() if key != "entropy_coef"}
    if left != right:
        raise ValueError("unequal paired training configuration")

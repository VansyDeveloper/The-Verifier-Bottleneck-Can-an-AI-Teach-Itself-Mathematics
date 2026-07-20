"""Empirical effective support and Barannikov exploration mass."""

import math
from collections import Counter

from modcomp.checker import extract_answer


def outcome_class(completion, p):
    """Map a completion to its canonical answer class; None means unparseable."""
    answer = extract_answer(completion)
    return None if answer is None else (answer[0] % p, answer[1] % p)


def novelty_metrics(reference, candidates, gold, resolution_m):
    """Estimate epsilon^(1/m) for one problem from independent sample batches.

    ``reference`` estimates the current policy pi. ``candidates`` come from the
    intervention rho. Both contain outcome classes returned by ``outcome_class``.
    """
    if not reference or not candidates:
        raise ValueError("reference and candidates must be non-empty")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    min_count = math.ceil(len(reference) / resolution_m)
    support = {u for u, count in Counter(reference).items() if count >= min_count}
    novel = [u not in support for u in candidates]
    parseable = [u is not None for u in candidates]
    correct = [u == gold for u in candidates]
    novel_count = sum(novel)
    novel_parseable_count = sum(n and p for n, p in zip(novel, parseable))
    novel_correct_count = sum(n and c for n, c in zip(novel, correct))

    return {
        "epsilon_raw": novel_count / len(candidates),
        "epsilon_parseable": novel_parseable_count / len(candidates),
        "epsilon_correct": novel_correct_count / len(candidates),
        "novel_precision": (
            novel_correct_count / novel_count if novel_count else None
        ),
        "novel_parseable_precision": (
            novel_correct_count / novel_parseable_count if novel_parseable_count else None
        ),
        "parse_rate": sum(parseable) / len(candidates),
        "pass@1": sum(correct) / len(candidates),
        "pass@k": float(any(correct)),
        "effective_support_size": len(support),
        "novel_parseable_classes": len(
            {u for u, n in zip(candidates, novel) if n and u is not None}
        ),
        "candidate_count": len(candidates),
        "novel_count": novel_count,
        "novel_parseable_count": novel_parseable_count,
        "novel_correct_count": novel_correct_count,
    }

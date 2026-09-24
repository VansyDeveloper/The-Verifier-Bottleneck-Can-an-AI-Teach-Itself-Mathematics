from __future__ import annotations

from collections import Counter
from math import comb, log
from typing import Iterable, Sequence


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if n < 0 or c < 0 or c > n or k < 1:
        raise ValueError("invalid n, c, or k")
    if n < k:
        raise ValueError("n must be at least k")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def empirical_pass_at_k(correct: Sequence[bool], k: int) -> float:
    if k < 1 or len(correct) < k:
        raise ValueError("need at least k candidates")
    return float(any(correct[:k]))


def signature_entropy(signatures: Iterable[tuple[str, ...]]) -> float:
    counts = Counter(signatures)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * log(count / total) for count in counts.values())

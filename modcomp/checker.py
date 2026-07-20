"""Noisy verifier with dials (alpha, beta).

``alpha = P(accept | correct)`` and ``beta = P(accept | wrong)``.  The primary
experiment keys one deterministic uniform draw by a unique rollout event id.
That makes reruns reproducible without making the verdict a property of the
answer text.  ``noisy_verdict`` is retained only for the legacy, answer-keyed
ablation.
"""

import hashlib
import re

# First "Answer: A=..., B=..." in the completion. First, not last: with a
# few-shot prompt the model often keeps going and invents new problems after
# answering the real one.
ANSWER_RE = re.compile(r"Answer:\s*A\s*=\s*(-?\d+)\s*,\s*B\s*=\s*(-?\d+)")


def extract_answer(completion):
    m = ANSWER_RE.search(completion)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def is_correct(completion, p, target_A, target_B):
    ans = extract_answer(completion)
    if ans is None:
        return False
    return ans[0] % p == target_A and ans[1] % p == target_B


def _unit_hash(*parts):
    """Deterministic uniform [0,1) from the given strings."""
    h = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def noisy_verdict_from_event(correct, event_id, alpha, beta):
    """Return ``(accepted, u)`` for one unique rollout event.

    ``event_id`` must be assigned before inspecting the completion and must be
    unique per rollout. Reusing it is intentionally idempotent; changing only
    the rollout id produces a fresh deterministic draw.
    """
    if not event_id:
        raise ValueError("event_id must be non-empty")
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError("alpha and beta must lie in [0, 1]")
    u = _unit_hash(str(event_id))
    return u < (alpha if correct else beta), u


def noisy_verdict(completion, prompt, p, target_A, target_B, alpha, beta, seed=0):
    """Legacy answer-keyed noise; not iid when a completion is repeated."""
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError("alpha and beta must lie in [0, 1]")
    correct = is_correct(completion, p, target_A, target_B)
    u = _unit_hash(str(seed), prompt, completion)
    accepted = u < (alpha if correct else beta)
    return accepted, correct

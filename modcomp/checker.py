"""Noisy verifier with dials (alpha, beta).

alpha = P(accept | answer correct), beta = P(accept | answer wrong).
alpha=1, beta=0 is the perfect checker; alpha=beta carries zero information.
Noise is deterministic given (completion, prompt, seed) so a re-run reproduces
the same verdicts.
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


def noisy_verdict(completion, prompt, p, target_A, target_B, alpha, beta, seed=0):
    """Returns (accepted, correct). Reward for RL is float(accepted)."""
    correct = is_correct(completion, p, target_A, target_B)
    u = _unit_hash(str(seed), prompt, completion)
    accepted = u < (alpha if correct else beta)
    return accepted, correct

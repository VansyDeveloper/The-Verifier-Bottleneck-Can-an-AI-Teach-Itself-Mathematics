"""Generator for modular linear-function composition tasks.

Task: given g_i(x) = a_i * x + b_i over F_p, compute the composition
h(x) = g_k(g_{k-1}(...g_1(x)...)) = A*x + B (mod p) and report (A, B).
"""

import random

# Small primes keep the arithmetic in a band where the base 1.7B model already
# has a nonzero, trainable pass@1 (~0.25-0.45 at k=2) with clear pass@k headroom
# to sharpen into. Larger primes push base pass@1 to ~0, leaving GRPO no signal
# *from the base model* -- but a curriculum that resumes from a checkpoint can
# climb a ceiling the base could not reach cold (see curriculum.sh).
#
# Train and eval primes are disjoint at every ceiling: EVAL_PRIMES is a fixed
# held-out subset spread across the whole range, so eval always probes prime-level
# generalization, not memorized coefficient tables. `prime_max` slices both pools
# to a difficulty ceiling; EVAL_PRIMES is chosen so at least one held-out prime
# survives even the smallest ceilings.
EVAL_PRIMES = [11, 17, 31, 41, 59, 83]
TRAIN_PRIMES = [5, 7, 13, 19, 23, 29, 37, 43, 47, 53, 61, 67, 71, 73, 79, 89, 97]


def primes_for(split, prime_max):
    pool = TRAIN_PRIMES if split == "train" else EVAL_PRIMES
    sel = [p for p in pool if p <= prime_max]
    if not sel:
        raise ValueError(f"no {split} primes <= {prime_max}; min is {min(pool)}")
    return sel


# Verbose style: rewrite the full nested expression each step. Faithful to the
# algebra but O(k^2) tokens -- a depth-10 solve overflows a 512-token budget, so
# it is only usable for shallow k. See FEWSHOT_COMPACT for deep compositions.
FEWSHOT = """Below are problems about composing linear functions modulo a prime p.
For each problem, compute h(x) = g_k(...g_2(g_1(x))...) in the form h(x) = A*x + B (mod p), with 0 <= A < p and 0 <= B < p.

Problem:
p = 13
g_1(x) = 4*x + 7
g_2(x) = 11*x + 2
Solution:
g_2(g_1(x)) = 11*(4*x + 7) + 2 = 44*x + 77 + 2 = 44*x + 79.
44 mod 13 = 5, 79 mod 13 = 1.
Answer: A=5, B=1

Problem:
p = 29
g_1(x) = 17*x + 20
g_2(x) = 3*x + 8
g_3(x) = 12*x + 25
Solution:
g_2(g_1(x)) = 3*(17*x + 20) + 8 = 51*x + 68 = 22*x + 10 (mod 29).
g_3(g_2(g_1(x))) = 12*(22*x + 10) + 25 = 264*x + 145 = 3*x + 0 (mod 29).
Answer: A=3, B=0

Problem:
"""


# Compact style: carry a running accumulator (A, B) and update it one function at
# a time, A<-a*A mod p, B<-a*B+b mod p. O(k) short lines, so a depth-10 solve fits
# in a few hundred tokens. The checker only reads the final "Answer:" line, so the
# reasoning format is free to change.
FEWSHOT_COMPACT = """Below are problems about composing linear functions modulo a prime p.
Compute h(x) = g_k(...g_2(g_1(x))...) = A*x + B (mod p), with 0 <= A < p and 0 <= B < p.
Start from (A, B) = (1, 0) [the identity] and fold in g_1, then g_2, ..., updating
A <- (a*A) mod p and B <- (a*B + b) mod p at each step.

Problem:
p = 13
g_1(x) = 4*x + 7
g_2(x) = 11*x + 2
Solution:
Start: A=1, B=0
g_1: A=4*1=4, B=4*0+7=7 -> A=4, B=7
g_2: A=11*4=44=5, B=11*7+2=79=1 -> A=5, B=1
Answer: A=5, B=1

Problem:
p = 29
g_1(x) = 17*x + 20
g_2(x) = 3*x + 8
g_3(x) = 12*x + 25
Solution:
Start: A=1, B=0
g_1: A=17*1=17, B=17*0+20=20 -> A=17, B=20
g_2: A=3*17=51=22, B=3*20+8=68=10 -> A=22, B=10
g_3: A=12*22=264=3, B=12*10+25=145=0 -> A=3, B=0
Answer: A=3, B=0

Problem:
"""


def compose(funcs, p):
    """funcs = [(a_1, b_1), ...] applied in order g_1 first. Returns (A, B)."""
    A, B = 1, 0
    for a, b in funcs:
        A = (a * A) % p
        B = (a * B + b) % p
    return A, B


def make_problem(rng, primes, k_min=2, k_max=2, style="verbose"):
    if style not in {"verbose", "compact"}:
        raise ValueError(f"unknown prompt style: {style}")
    p = rng.choice(primes)
    k = rng.randint(k_min, k_max)
    funcs = [(rng.randint(1, p - 1), rng.randint(0, p - 1)) for _ in range(k)]
    A, B = compose(funcs, p)
    lines = [f"p = {p}"] + [f"g_{i + 1}(x) = {a}*x + {b}" for i, (a, b) in enumerate(funcs)]
    header = FEWSHOT_COMPACT if style == "compact" else FEWSHOT
    prompt = header + "\n".join(lines) + "\nSolution:\n"
    return {"prompt": prompt, "p": p, "target_A": A, "target_B": B, "k": k}


def make_dataset(n, seed, split="train", k_min=2, k_max=2, prime_max=97, style="verbose"):
    if n < 0:
        raise ValueError("n must be non-negative")
    if k_min < 1 or k_min > k_max:
        raise ValueError("k must satisfy 1 <= k_min <= k_max")
    primes = primes_for(split, prime_max)
    max_unique = sum(((p - 1) * p) ** k for p in primes for k in range(k_min, k_max + 1))
    if n > max_unique:
        raise ValueError(f"requested {n} unique problems, but only {max_unique} exist")
    rng = random.Random(seed)
    seen = set()
    rows = []
    while len(rows) < n:
        row = make_problem(rng, primes, k_min, k_max, style)
        key = row["prompt"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows

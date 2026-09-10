from __future__ import annotations

import itertools
import math
from fractions import Fraction
from statistics import fmean, stdev
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t


def _ordered_matrix(control_by_seed: Mapping[int, Mapping[str, float]],
                    distill_by_seed: Mapping[int, Mapping[str, float]]) -> tuple[list[int], list[str], np.ndarray]:
    seeds = sorted(control_by_seed)
    if seeds != sorted(distill_by_seed) or len(seeds) != 6:
        raise ValueError("exactly the same six seeds are required")
    task_ids = sorted(control_by_seed[seeds[0]])
    if not task_ids:
        raise ValueError("task matrix is empty")
    matrix = np.empty((6, len(task_ids)), dtype=np.float64)
    expected = set(task_ids)
    for row_index, seed in enumerate(seeds):
        if set(control_by_seed[seed]) != expected or set(distill_by_seed[seed]) != expected:
            raise ValueError(f"seed {seed} task set differs")
        for column_index, task_id in enumerate(task_ids):
            control = float(control_by_seed[seed][task_id])
            distill = float(distill_by_seed[seed][task_id])
            if control not in (0.0, 1.0) or distill not in (0.0, 1.0):
                raise ValueError("Hit@32 rows must be binary")
            matrix[row_index, column_index] = distill - control
    return seeds, task_ids, matrix


def per_seed_binary_counts(seeds: Sequence[int], task_ids: Sequence[str], matrix: np.ndarray,
                           control_by_seed: Mapping[int, Mapping[str, float]],
                           distill_by_seed: Mapping[int, Mapping[str, float]]) -> list[dict]:
    rows = []
    for row_index, seed in enumerate(seeds):
        control = np.asarray([control_by_seed[seed][task_id] for task_id in task_ids], dtype=np.int64)
        distill = np.asarray([distill_by_seed[seed][task_id] for task_id in task_ids], dtype=np.int64)
        both = int(np.sum((control == 1) & (distill == 1)))
        distill_only = int(np.sum((control == 0) & (distill == 1)))
        control_only = int(np.sum((control == 1) & (distill == 0)))
        neither = int(np.sum((control == 0) & (distill == 0)))
        numerator = distill_only - control_only
        denominator = len(task_ids)
        delta = float(matrix[row_index].mean())
        if numerator / denominator != delta:
            raise RuntimeError("integer and floating delta disagree")
        rows.append({"replicate_label": int(seed), "both_hit": both, "distill_only": distill_only,
                     "control_only": control_only, "neither": neither,
                     "control_hits": int(control.sum()), "distill_hits": int(distill.sum()),
                     "delta_numerator": numerator, "delta_denominator": denominator,
                     "delta": delta, "delta_pp": 100.0 * delta})
    return rows


def seed_t_statistics(deltas: Sequence[float]) -> dict:
    values = [float(value) for value in deltas]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("six finite seed deltas are required")
    mean = fmean(values)
    sd = stdev(values)
    se = sd / math.sqrt(6)
    if se == 0.0:
        if mean == 0.0:
            statistic, marker, p_value = 0.0, None, 1.0
        else:
            statistic, marker, p_value = None, ("positive_infinity" if mean > 0 else "negative_infinity"), 0.0
        interval = [mean, mean]
    else:
        statistic = mean / se
        marker = None
        p_value = float(2.0 * student_t.sf(abs(statistic), 5))
        half_width = float(student_t.ppf(0.975, 5)) * se
        interval = [mean - half_width, mean + half_width]
    return {"n": 6, "df": 5, "mean_delta": mean, "mean_delta_pp": 100.0 * mean,
            "sample_sd": sd, "standard_error": se, "t": statistic, "t_nonfinite": marker,
            "p_two_sided": p_value, "t_ci95": interval,
            "positive_seed_count": sum(value > 0.0 for value in values)}


def exact_sign_flip(numerators: Sequence[int], denominator: int) -> dict:
    nums = [int(value) for value in numerators]
    if len(nums) != 6 or denominator <= 0:
        raise ValueError("six integer numerators and a positive denominator are required")
    observed = abs(sum(nums))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=6):
        if abs(sum(sign * value for sign, value in zip(signs, nums))) >= observed:
            extreme += 1
    fraction = Fraction(extreme, 64)
    return {"statistic": "absolute_mean_seed_delta", "alternative": "two_sided",
            "ties": "included", "extreme_count": extreme, "permutations": 64,
            "p_numerator": fraction.numerator, "p_denominator": fraction.denominator,
            "p_two_sided": float(fraction), "minimum_attainable_p": 0.03125,
            "gating": False}


def crossed_seed_task_bootstrap(matrix: np.ndarray, repetitions: int = 20000,
                                seed: int = 20260727) -> tuple[dict, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 6 or values.shape[1] == 0:
        raise ValueError("bootstrap matrix must have shape (6, N)")
    if repetitions < 2:
        raise ValueError("repetitions must be >=2")
    rng = np.random.Generator(np.random.PCG64(seed))
    n_tasks = values.shape[1]
    replicates = np.empty(repetitions, dtype=np.float64)
    batch = min(256, repetitions)
    for start in range(0, repetitions, batch):
        stop = min(start + batch, repetitions)
        count = stop - start
        seed_draws = rng.integers(0, 6, size=(count, 6))
        task_draws = rng.integers(0, n_tasks, size=(count, n_tasks))
        for index in range(count):
            replicates[start + index] = values[np.ix_(seed_draws[index], task_draws[index])].mean()
    lower, upper = np.quantile(replicates, (0.025, 0.975), method="linear")
    receipt = {"algorithm": "crossed_seed_task_percentile_v1", "repetitions": repetitions,
               "rng": {"library": "numpy", "bit_generator": "PCG64", "seed": seed},
               "seed_draw_size": 6, "task_draw_size": n_tasks,
               "shared_task_draw_across_seeds": True, "equal_seed_weight": True,
               "dtype": "float64", "quantile_method": "linear", "quantiles": [0.025, 0.975],
               "observed_mean": float(values.mean()), "bootstrap_mean": float(replicates.mean()),
               "ci95": [float(lower), float(upper)]}
    return receipt, replicates


def holm(pvalues: Mapping[str, float]) -> list[dict]:
    ordered = sorted(((str(key), float(value)) for key, value in pvalues.items()), key=lambda item: (item[1], item[0]))
    if len(ordered) != 5 or any(not 0.0 <= value <= 1.0 for _, value in ordered):
        raise ValueError("the fixed Holm family must contain five valid p-values")
    running = 0.0
    rows = []
    for index, (endpoint, raw) in enumerate(ordered):
        running = max(running, (5 - index) * raw)
        adjusted = min(1.0, running)
        rows.append({"endpoint": endpoint, "raw_p": raw, "holm_adjusted_p": adjusted,
                     "reject_at_0p05": adjusted < 0.05})
    return rows


def analyze_primary(control_by_seed: Mapping[int, Mapping[str, float]],
                    distill_by_seed: Mapping[int, Mapping[str, float]],
                    repetitions: int = 20000, bootstrap_seed: int = 20260727) -> tuple[dict, np.ndarray]:
    seeds, task_ids, matrix = _ordered_matrix(control_by_seed, distill_by_seed)
    per_seed = per_seed_binary_counts(seeds, task_ids, matrix, control_by_seed, distill_by_seed)
    seed_stats = seed_t_statistics([row["delta"] for row in per_seed])
    sign_flip = exact_sign_flip([row["delta_numerator"] for row in per_seed], len(task_ids))
    bootstrap, replicates = crossed_seed_task_bootstrap(matrix, repetitions, bootstrap_seed)
    checks = {"mean_delta_ge_005": seed_stats["mean_delta"] >= 0.05,
              "t_ci_lower_gt_0": seed_stats["t_ci95"][0] > 0.0,
              "t_p_two_sided_lt_005": seed_stats["p_two_sided"] < 0.05,
              "positive_seeds_ge_5_of_6": seed_stats["positive_seed_count"] >= 5}
    return ({"endpoint": {"split": "final_a", "depth": 3, "metric": "hit@32",
                           "baseline": "atomic_control", "treatment": "composition_distill",
                           "n_tasks_per_seed": len(task_ids)},
             "per_seed": per_seed, "seed_statistics": seed_stats, "exact_sign_flip": sign_flip,
             "seed_task_bootstrap": bootstrap,
             "criteria": {"checks": checks, "composition_effect_pass": all(checks.values())}}, replicates)


from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, stdev
from typing import Any


HIT_KEYS = ("hit@1", "hit@8", "hit@16", "hit@32", "hit@64")
METRIC_KEYS = (*HIT_KEYS, "correct_mass", "best_rank", "mrr", "log_gap")
METRIC_ALIASES = {"mass": "correct_mass", "best-rank": "best_rank", "log-gap": "log_gap"}
DEFAULT_BOOTSTRAP_REPETITIONS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20_260_727


def _stable_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _ordered_mapping_items(values: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    return sorted(values.items(), key=lambda item: _stable_key(item[0]))


def _metric_name(metric: str) -> str:
    resolved = METRIC_ALIASES.get(metric, metric)
    if resolved not in METRIC_KEYS:
        raise ValueError(f"unsupported metric: {metric!r}")
    return resolved


def _task_id(row: Mapping[str, Any]) -> str:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("each metric row must have a non-empty string task_id")
    return task_id


def _row_seed(row: Mapping[str, Any], *, required: bool) -> Any | None:
    binding = row.get("binding")
    top_seed = row.get("seed") if "seed" in row else None
    bound_seed = binding.get("seed") if isinstance(binding, Mapping) and "seed" in binding else None
    if top_seed is not None and bound_seed is not None and top_seed != bound_seed:
        raise ValueError(f"task {_task_id(row)!r} has contradictory top-level and binding seed")
    if top_seed is not None:
        return top_seed
    if bound_seed is not None:
        return bound_seed
    if required:
        raise ValueError(f"task {_task_id(row)!r} is missing seed")
    return None


def _metric_value(row: Mapping[str, Any], metric: str, *, finite: bool = True) -> float:
    metric = _metric_name(metric)
    if metric not in row:
        raise ValueError(f"task {_task_id(row)!r} is missing metric {metric!r}")
    raw = row[metric]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"task {_task_id(row)!r} metric {metric!r} is not numeric")
    value = float(raw)
    if math.isnan(value) or (finite and not math.isfinite(value)):
        raise ValueError(f"task {_task_id(row)!r} metric {metric!r} must be finite")
    if metric in (*HIT_KEYS, "correct_mass", "mrr") and not 0.0 <= value <= 1.0:
        raise ValueError(f"task {_task_id(row)!r} metric {metric!r} must be in [0, 1]")
    if metric in HIT_KEYS and value not in (0.0, 1.0):
        raise ValueError(f"task {_task_id(row)!r} metric {metric!r} must be binary")
    if metric == "best_rank" and value < 1.0:
        raise ValueError(f"task {_task_id(row)!r} best_rank must be positive")
    if metric == "mrr" and value == 0.0:
        raise ValueError(f"task {_task_id(row)!r} mrr must be positive")
    return value


def _single_context(rows: Sequence[Mapping[str, Any]], field: str) -> Any | None:
    if field == "seed":
        present = [_row_seed(row, required=False) for row in rows]
        present = [value for value in present if value is not None]
    else:
        present = [row[field] for row in rows if field in row]
    if present and len(present) != len(rows):
        raise ValueError(f"field {field!r} must be present on either every row or no rows")
    values = set(present)
    if len(values) > 1:
        raise ValueError(f"strict task join requires one {field}; found {sorted(values, key=_stable_key)!r}")
    return next(iter(values)) if values else None


def _index_task_rows(rows: Iterable[Mapping[str, Any]], label: str) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"{label} rows are empty")
    index: dict[str, Mapping[str, Any]] = {}
    for row in materialized:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} contains a non-mapping row")
        task_id = _task_id(row)
        if task_id in index:
            raise ValueError(f"duplicate task_id {task_id!r} in {label}")
        index[task_id] = row
    return materialized, index


def strict_join_by_task_id(
    baseline_rows: Iterable[Mapping[str, Any]],
    treatment_rows: Iterable[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """Join one seed/split pair by task_id, rejecting duplicates and any missing task.

    Input order is intentionally ignored.  If seed or split metadata is present,
    both collections must describe the same single seed and split.
    """

    baseline, baseline_index = _index_task_rows(baseline_rows, "baseline")
    treatment, treatment_index = _index_task_rows(treatment_rows, "treatment")
    for field in ("seed", "split"):
        baseline_context = _single_context(baseline, field)
        treatment_context = _single_context(treatment, field)
        if baseline_context != treatment_context:
            raise ValueError(
                f"{field} mismatch: baseline={baseline_context!r}, treatment={treatment_context!r}"
            )

    baseline_ids = set(baseline_index)
    treatment_ids = set(treatment_index)
    if baseline_ids != treatment_ids:
        missing_in_treatment = sorted(baseline_ids - treatment_ids)
        missing_in_baseline = sorted(treatment_ids - baseline_ids)
        raise ValueError(
            "task_id sets differ: "
            f"missing_in_treatment={missing_in_treatment!r}, "
            f"missing_in_baseline={missing_in_baseline!r}"
        )
    joined = []
    for task_id in sorted(baseline_ids):
        baseline_row = baseline_index[task_id]
        treatment_row = treatment_index[task_id]
        for field in ("task_fingerprint", "split", "p", "depth"):
            baseline_value = baseline_row.get(field)
            treatment_value = treatment_row.get(field)
            if baseline_value != treatment_value:
                raise ValueError(
                    f"task {task_id!r} {field} mismatch: "
                    f"baseline={baseline_value!r}, treatment={treatment_value!r}"
                )
        joined.append((task_id, baseline_row, treatment_row))
    return joined


def aggregate_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the frozen ranking metrics with duplicate-row protection.

    Infinite log gaps are retained in the raw rows but excluded from their mean,
    matching the existing Stage 4 summary convention.  The finite count makes
    that exclusion explicit.
    """

    materialized = list(rows)
    if not materialized:
        raise ValueError("metric rows are empty")
    identities: set[tuple[Any, Any, Any, str]] = set()
    for row in materialized:
        if not isinstance(row, Mapping):
            raise ValueError("metric rows must be mappings")
        identity = (_row_seed(row, required=False), row.get("split"), row.get("branch"), _task_id(row))
        if identity in identities:
            raise ValueError(f"duplicate metric row identity: {identity!r}")
        identities.add(identity)

    result: dict[str, Any] = {"n_tasks": len(materialized), "finite_counts": {}}
    for metric in METRIC_KEYS:
        values = [_metric_value(row, metric, finite=False) for row in materialized]
        finite_values = [value for value in values if math.isfinite(value)]
        result[metric] = fmean(finite_values) if finite_values else None
        result["finite_counts"][metric] = len(finite_values)
    return result


def paired_task_differences(
    baseline_rows: Iterable[Mapping[str, Any]],
    treatment_rows: Iterable[Mapping[str, Any]],
    metric: str = "hit@32",
) -> dict[str, float]:
    """Return treatment-minus-baseline differences keyed by task_id."""

    metric = _metric_name(metric)
    result: dict[str, float] = {}
    for task_id, baseline, treatment in strict_join_by_task_id(baseline_rows, treatment_rows):
        result[task_id] = _metric_value(treatment, metric) - _metric_value(baseline, metric)
    return result


def _rows_by_seed(
    rows: Iterable[Mapping[str, Any]], label: str, split: str | None
) -> dict[Any, list[Mapping[str, Any]]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    seen_splits: set[Any] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} contains a non-mapping row")
        row_seed = _row_seed(row, required=True)
        row_split = row.get("split")
        if split is not None and row_split != split:
            continue
        seen_splits.add(row_split)
        grouped[row_seed].append(row)
    if not grouped:
        suffix = f" for split {split!r}" if split is not None else ""
        raise ValueError(f"{label} rows are empty{suffix}")
    if split is None and len(seen_splits) > 1:
        raise ValueError(f"{label} contains multiple splits; select one explicitly")
    return dict(grouped)


def paired_seed_task_differences(
    baseline_rows: Iterable[Mapping[str, Any]],
    treatment_rows: Iterable[Mapping[str, Any]],
    metric: str = "hit@32",
    *,
    split: str | None = None,
) -> dict[Any, dict[str, float]]:
    """Strictly pair flat branch rows within every seed and return task deltas."""

    baseline_by_seed = _rows_by_seed(baseline_rows, "baseline", split)
    treatment_by_seed = _rows_by_seed(treatment_rows, "treatment", split)
    baseline_seeds = set(baseline_by_seed)
    treatment_seeds = set(treatment_by_seed)
    if baseline_seeds != treatment_seeds:
        raise ValueError(
            "seed sets differ: "
            f"missing_in_treatment={sorted(baseline_seeds - treatment_seeds, key=_stable_key)!r}, "
            f"missing_in_baseline={sorted(treatment_seeds - baseline_seeds, key=_stable_key)!r}"
        )
    result: dict[Any, dict[str, float]] = {}
    for seed in sorted(baseline_seeds, key=_stable_key):
        result[seed] = paired_task_differences(
            baseline_by_seed[seed], treatment_by_seed[seed], metric
        )
    return result


def aggregate_seed_differences(seed_task_differences: Mapping[Any, Mapping[str, float]]) -> dict[Any, float]:
    if not seed_task_differences:
        raise ValueError("seed task differences are empty")
    result: dict[Any, float] = {}
    for seed, task_differences in _ordered_mapping_items(seed_task_differences):
        if not isinstance(task_differences, Mapping) or not task_differences:
            raise ValueError(f"seed {seed!r} has no task differences")
        values = [float(value) for _, value in _ordered_mapping_items(task_differences)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"seed {seed!r} contains a non-finite task difference")
        result[seed] = fmean(values)
    return result


def _difference_values(differences: Mapping[Any, float] | Sequence[float]) -> list[float]:
    if isinstance(differences, Mapping):
        values = [float(value) for _, value in _ordered_mapping_items(differences)]
    elif isinstance(differences, Sequence) and not isinstance(differences, (str, bytes)):
        values = [float(value) for value in differences]
    else:
        raise ValueError("differences must be a mapping or a numeric sequence")
    if not values:
        raise ValueError("differences are empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("differences must be finite")
    return values


def seed_statistics(
    differences: Mapping[Any, float] | Sequence[float], alpha: float = 0.05
) -> dict[str, Any]:
    """Student one-sample t inference over independent seed-level deltas."""

    values = _difference_values(differences)
    if len(values) < 2:
        raise ValueError("Student inference requires at least two seed differences")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:  # pragma: no cover - analysis environment includes scipy
        raise RuntimeError("scipy is required for exact Student t inference") from exc

    n = len(values)
    mean_delta = fmean(values)
    standard_deviation = stdev(values)
    standard_error = standard_deviation / math.sqrt(n)
    degrees_of_freedom = n - 1
    critical = float(student_t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    if standard_error == 0.0:
        if mean_delta == 0.0:
            t_statistic, t_nonfinite, p_two_sided = 0.0, None, 1.0
        else:
            # Strict JSON cannot represent infinities. Preserve the exact
            # statistical case explicitly while keeping all serialized values finite.
            t_statistic = None
            t_nonfinite = "positive_infinity" if mean_delta > 0.0 else "negative_infinity"
            p_two_sided = 0.0
        interval = [mean_delta, mean_delta]
    else:
        t_statistic = mean_delta / standard_error
        t_nonfinite = None
        p_two_sided = float(2.0 * student_t.sf(abs(t_statistic), degrees_of_freedom))
        half_width = critical * standard_error
        interval = [mean_delta - half_width, mean_delta + half_width]
    return {
        "n": n,
        "mean_delta": mean_delta,
        "degrees_of_freedom": degrees_of_freedom,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "t": t_statistic,
        "t_nonfinite": t_nonfinite,
        "zero_variance": standard_error == 0.0,
        "t_ci95": interval,
        "p_two_sided": p_two_sided,
        "positive_seeds": sum(value > 0.0 for value in values),
        "positive_ge_5pp": sum(value >= 0.05 for value in values),
    }


def six_seed_statistics(differences: Mapping[Any, float] | Sequence[float]) -> dict[str, Any]:
    values = _difference_values(differences)
    if len(values) != 6:
        raise ValueError(f"confirmation inference requires exactly six seeds, got {len(values)}")
    return seed_statistics(values)


def exact_sign_flip(differences: Mapping[Any, float] | Sequence[float]) -> float:
    """Return the deterministic exact two-sided sign-flip p-value."""

    values = _difference_values(differences)
    if len(values) > 20:
        raise ValueError("exact sign-flip is restricted to at most 20 seeds")
    observed = abs(fmean(values))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted_sum = math.fsum(signs[index] * values[index] for index in range(len(values)))
        if abs(permuted_sum / len(values)) >= observed - 1e-15:
            extreme += 1
    return extreme / (2 ** len(values))


def hierarchical_bootstrap(
    seed_tasks: Mapping[Any, Mapping[str, float] | Sequence[float]],
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Two-level seed x task bootstrap with equal seed weighting."""

    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if not seed_tasks:
        raise ValueError("seed task differences are empty")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - analysis environment includes numpy
        raise RuntimeError("numpy is required for the hierarchical bootstrap") from exc

    arrays: list[Any] = []
    seed_labels: list[Any] = []
    for seed_label, tasks in _ordered_mapping_items(seed_tasks):
        if isinstance(tasks, Mapping):
            values = [float(value) for _, value in _ordered_mapping_items(tasks)]
        elif isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
            values = [float(value) for value in tasks]
        else:
            raise ValueError(f"seed {seed_label!r} tasks must be a mapping or sequence")
        if not values:
            raise ValueError(f"seed {seed_label!r} has no task differences")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"seed {seed_label!r} contains a non-finite task difference")
        arrays.append(np.asarray(values, dtype=np.float64))
        seed_labels.append(seed_label)

    rng = np.random.default_rng(seed)
    n_seeds = len(arrays)
    bootstrap_values = np.empty(repetitions, dtype=np.float64)
    batch_size = min(512, repetitions)
    for start in range(0, repetitions, batch_size):
        stop = min(repetitions, start + batch_size)
        count = stop - start
        selected_seeds = rng.integers(0, n_seeds, size=(count, n_seeds))
        replicate_sums = np.zeros(count, dtype=np.float64)
        for slot in range(n_seeds):
            selected_for_slot = selected_seeds[:, slot]
            slot_means = np.empty(count, dtype=np.float64)
            for seed_index, task_values in enumerate(arrays):
                positions = np.flatnonzero(selected_for_slot == seed_index)
                if positions.size == 0:
                    continue
                task_indices = rng.integers(
                    0, task_values.size, size=(positions.size, task_values.size)
                )
                slot_means[positions] = task_values[task_indices].mean(axis=1)
            replicate_sums += slot_means
        bootstrap_values[start:stop] = replicate_sums / n_seeds

    lower, upper = np.quantile(bootstrap_values, (0.025, 0.975), method="linear")
    observed = fmean(float(values.mean()) for values in arrays)
    bootstrap_mean = float(bootstrap_values.mean())
    return {
        "repetitions": repetitions,
        "seed": seed,
        "n_seeds": n_seeds,
        "observed_mean": observed,
        "mean": bootstrap_mean,
        "bootstrap_mean": bootstrap_mean,
        "ci95": [float(lower), float(upper)],
        "seed_labels": seed_labels,
    }


def holm(pvalues: Mapping[Any, float]) -> dict[Any, float]:
    """Holm step-down family-wise p-value adjustment."""

    if not pvalues:
        return {}
    validated: list[tuple[Any, float]] = []
    for label, raw in pvalues.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"p-value for {label!r} is not numeric")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"p-value for {label!r} must be finite and in [0, 1]")
        validated.append((label, value))
    validated.sort(key=lambda item: (item[1], _stable_key(item[0])))
    family_size = len(validated)
    running = 0.0
    adjusted: dict[Any, float] = {}
    for rank, (label, value) in enumerate(validated):
        running = max(running, (family_size - rank) * value)
        adjusted[label] = min(1.0, running)
    return adjusted


def confirmation_criteria(
    differences: Mapping[Any, float] | Sequence[float],
    *,
    forgetting: float,
    mass_growth: float,
) -> dict[str, Any]:
    """Evaluate all preregistered conditions for one final split (A or B)."""

    if not math.isfinite(float(forgetting)) or not math.isfinite(float(mass_growth)):
        raise ValueError("forgetting and mass_growth must be finite")
    statistics = six_seed_statistics(differences)
    checks = {
        "mean_delta_ge_005": statistics["mean_delta"] >= 0.05,
        "t_ci_lower_gt_0": statistics["t_ci95"][0] > 0.0,
        "t_p_two_sided_lt_005": statistics["p_two_sided"] < 0.05,
        "positive_seeds_ge_5_of_6": statistics["positive_seeds"] >= 5,
        "forgetting_le_002": float(forgetting) <= 0.02,
        "mass_growth_gt_0": float(mass_growth) > 0.0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "mean_delta": 0.05,
            "ci_lower_strictly_above": 0.0,
            "p_two_sided_strictly_below": 0.05,
            "positive_seeds": 5,
            "forgetting": 0.02,
            "mass_growth_strictly_above": 0.0,
        },
    }


def criteria_a_b(
    differences_a: Mapping[Any, float] | Sequence[float],
    differences_b: Mapping[Any, float] | Sequence[float],
    *,
    forgetting_a: float,
    mass_growth_a: float,
    forgetting_b: float,
    mass_growth_b: float,
) -> dict[str, Any]:
    """Apply the complete confirmation gate independently to final A and B."""

    result_a = confirmation_criteria(
        differences_a, forgetting=forgetting_a, mass_growth=mass_growth_a
    )
    result_b = confirmation_criteria(
        differences_b, forgetting=forgetting_b, mass_growth=mass_growth_b
    )
    return {"A": result_a, "B": result_b, "pass_both": result_a["pass"] and result_b["pass"]}


def analyze_confirmation(
    baseline_rows: Iterable[Mapping[str, Any]],
    treatment_rows: Iterable[Mapping[str, Any]],
    *,
    forgetting: float,
    split: str | None = None,
    primary_metric: str = "hit@32",
    secondary_pvalues: Mapping[Any, float] | None = None,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the complete six-seed confirmation analysis from flat paired rows."""

    baseline = list(baseline_rows)
    treatment = list(treatment_rows)
    primary_task_differences = paired_seed_task_differences(
        baseline, treatment, primary_metric, split=split
    )
    primary_seed_differences = aggregate_seed_differences(primary_task_differences)
    statistics = six_seed_statistics(primary_seed_differences)
    mass_task_differences = paired_seed_task_differences(
        baseline, treatment, "correct_mass", split=split
    )
    mass_seed_differences = aggregate_seed_differences(mass_task_differences)
    if set(primary_seed_differences) != set(mass_seed_differences):
        raise ValueError("primary and correct-mass seed sets differ")
    mass_growth = fmean(mass_seed_differences.values())
    criteria = confirmation_criteria(
        primary_seed_differences, forgetting=forgetting, mass_growth=mass_growth
    )
    return {
        "metric": _metric_name(primary_metric),
        "seed_differences": primary_seed_differences,
        "seed_statistics": statistics,
        "exact_sign_flip_p_two_sided": exact_sign_flip(primary_seed_differences),
        "bootstrap": hierarchical_bootstrap(
            primary_task_differences,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        ),
        "mass_seed_differences": mass_seed_differences,
        "mass_growth": mass_growth,
        "forgetting": float(forgetting),
        "criteria": criteria,
        "secondary_holm": holm(secondary_pvalues or {}),
    }


# Explicit aliases used by reporting code and earlier Stage 4 terminology.
join_by_task_id = strict_join_by_task_id
two_level_bootstrap = hierarchical_bootstrap
holm_correction = holm

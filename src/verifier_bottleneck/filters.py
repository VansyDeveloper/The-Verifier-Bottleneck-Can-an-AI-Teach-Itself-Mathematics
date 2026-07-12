import random


def exact_filter(samples: list[dict]) -> list[dict]:
    return [sample for sample in samples if sample.get("is_correct", False)]


def rate_matched_random_filter(
    samples: list[dict],
    n_selected: int,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)

    if n_selected >= len(samples):
        return list(samples)

    return rng.sample(samples, n_selected)
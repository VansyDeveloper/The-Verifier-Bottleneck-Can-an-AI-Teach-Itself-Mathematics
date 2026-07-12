def pass_at_k(correct_flags: list[bool], k: int) -> bool:
    return any(correct_flags[:k])


def accuracy_at_1(correct_flags: list[bool]) -> bool:
    return bool(correct_flags[0]) if correct_flags else False


def mean_bool(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_passk(all_flags: list[list[bool]], ks: list[int]) -> dict:
    metrics = {}

    metrics["pass@1"] = mean_bool([
        accuracy_at_1(flags)
        for flags in all_flags
    ])

    for k in ks:
        metrics[f"pass@{k}"] = mean_bool([
            pass_at_k(flags, k)
            for flags in all_flags
        ])

    return metrics
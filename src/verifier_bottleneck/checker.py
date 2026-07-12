import re

MOD = 101


def parse_pair(text: str) -> tuple[int, int] | None:
    matches = re.findall(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", text)

    if not matches:
        return None

    a_raw, b_raw = matches[-1]
    return int(a_raw) % MOD, int(b_raw) % MOD


def check_answer(text: str, gold_a: int, gold_b: int) -> bool:
    parsed = parse_pair(text)

    if parsed is None:
        return False

    pred_a, pred_b = parsed
    return pred_a == gold_a % MOD and pred_b == gold_b % MOD
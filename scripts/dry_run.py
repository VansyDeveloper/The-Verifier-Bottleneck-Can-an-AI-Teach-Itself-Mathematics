import json
import random
from pathlib import Path

from verifier_bottleneck.checker import check_answer
from verifier_bottleneck.manifests import read_jsonl
from verifier_bottleneck.metrics import summarize_passk


def mock_generate(task: dict, rng: random.Random, p_correct: float = 0.35) -> str:
    gold_a = task["answer"]["A"]
    gold_b = task["answer"]["B"]

    if rng.random() < p_correct:
        return f"We compute the composition. Final answer: ({gold_a}, {gold_b})."

    wrong_a = rng.randint(0, 100)
    wrong_b = rng.randint(0, 100)
    return f"We compute the composition. Final answer: ({wrong_a}, {wrong_b})."


def main() -> None:
    rng = random.Random(42)

    tasks = read_jsonl("data/manifests/heldout_l2.jsonl")[:50]
    k = 16

    all_flags = []
    rows = []

    for task in tasks:
        samples = []

        for sample_id in range(k):
            text = mock_generate(task, rng)
            is_correct = check_answer(
                text,
                task["answer"]["A"],
                task["answer"]["B"],
            )

            samples.append({
                "sample_id": sample_id,
                "text": text,
                "is_correct": is_correct,
            })

        flags = [sample["is_correct"] for sample in samples]
        all_flags.append(flags)

        rows.append({
            "task_id": task["id"],
            "gold_answer": task["answer_text"],
            "samples": samples,
        })

    metrics = summarize_passk(all_flags, ks=[1, 8, 16])

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with (metrics_dir / "dry_run_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pools_dir = Path("outputs/pools")
    pools_dir.mkdir(parents=True, exist_ok=True)

    with (pools_dir / "dry_run_pool.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
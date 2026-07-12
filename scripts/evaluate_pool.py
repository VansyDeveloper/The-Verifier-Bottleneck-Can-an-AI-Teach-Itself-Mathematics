import argparse
import json
from pathlib import Path

from verifier_bottleneck.metrics import summarize_passk


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def evaluate_pool(rows: list[dict], ks: list[int]) -> dict:
    all_flags = []

    total_samples = 0
    total_correct_samples = 0

    for row in rows:
        samples = row.get("samples", [])
        flags = []

        for sample in samples:
            is_correct = bool(sample.get("is_correct", False))
            flags.append(is_correct)

            total_samples += 1
            total_correct_samples += int(is_correct)

        all_flags.append(flags)

    metrics = summarize_passk(all_flags, ks=ks)

    metrics["n_tasks"] = len(rows)
    metrics["total_samples"] = total_samples
    metrics["avg_samples_per_task"] = (
        total_samples / len(rows) if rows else 0.0
    )
    metrics["sample_accuracy"] = (
        total_correct_samples / total_samples if total_samples else 0.0
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pool_path",
        type=str,
        help="Path to JSONL pool file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/metrics/evaluated_pool_metrics.json",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 8, 16],
    )

    args = parser.parse_args()

    rows = read_jsonl(args.pool_path)
    metrics = evaluate_pool(rows, ks=args.ks)

    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
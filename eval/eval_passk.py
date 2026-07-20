"""pass@k evaluation, optional sharding, and shard merging.

Examples:
  python eval/eval_passk.py --model runs/model/final --k 64
  python eval/eval_passk.py --model M --k 1024 --shard 0 --num-shards 8 --out part0.json
  python eval/eval_passk.py --merge part0.json part1.json --out merged.json
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from modcomp.checker import is_correct
from modcomp.gen import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--split", default="eval", choices=["train", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument(
        "--prime-max",
        type=int,
        default=97,
        help="difficulty ceiling; MUST match the checkpoint's training range",
    )
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--style", choices=["verbose", "compact"], default="verbose")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", nargs="+", metavar="PART")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args.merge, args.out)
        return
    if not args.model:
        ap.error("--model is required unless --merge is used")
    if not 0 <= args.shard < args.num_shards:
        ap.error("--shard must satisfy 0 <= shard < num-shards")
    if args.num_shards > 1 and not args.out:
        ap.error("--out is required for sharded evaluation")

    from vllm import LLM, SamplingParams

    all_rows = make_dataset(
        args.n_problems,
        seed=args.seed + 10_000,
        split=args.split,
        k_min=args.k_min,
        k_max=args.k_max,
        prime_max=args.prime_max,
        style=args.style,
    )
    indexed_rows = list(enumerate(all_rows))[args.shard :: args.num_shards]
    rows = [row for _, row in indexed_rows]
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85)
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outs = llm.generate([r["prompt"] for r in rows], sp)

    n_correct_total, n_samples_total, solved = 0, 0, 0
    per_problem = []
    for (problem_index, r), o in zip(indexed_rows, outs):
        c = sum(is_correct(s.text, r["p"], r["target_A"], r["target_B"]) for s in o.outputs)
        n_correct_total += c
        n_samples_total += len(o.outputs)
        solved += c > 0
        per_problem.append(
            {
                "problem_index": problem_index,
                "p": r["p"],
                "k_funcs": r["k"],
                "n_correct": c,
                "n": len(o.outputs),
            }
        )

    result = {
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "temperature": args.temperature,
        "k": args.k,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "prime_max": args.prime_max,
        "max_tokens": args.max_tokens,
        "style": args.style,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_problems": len(per_problem),
        "pass@1": n_correct_total / n_samples_total,
        f"pass@{args.k}": solved / len(per_problem),
        "per_problem": per_problem,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_problem"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


CONFIG_FIELDS = (
    "model",
    "split",
    "seed",
    "temperature",
    "k",
    "k_min",
    "k_max",
    "prime_max",
    "max_tokens",
    "style",
    "num_shards",
)


def merge_shards(paths, out):
    if not out:
        raise SystemExit("--out is required with --merge")

    parts = []
    for path in paths:
        with open(path) as f:
            parts.append(json.load(f))

    reference = {field: parts[0].get(field) for field in CONFIG_FIELDS}
    for path, part in zip(paths[1:], parts[1:]):
        actual = {field: part.get(field) for field in CONFIG_FIELDS}
        if actual != reference:
            raise ValueError(f"incompatible shard config in {path}")

    expected_shards = set(range(reference["num_shards"]))
    actual_shards = {part["shard"] for part in parts}
    if actual_shards != expected_shards:
        raise ValueError(f"expected shards {sorted(expected_shards)}, got {sorted(actual_shards)}")

    per_problem = sorted(
        (row for part in parts for row in part["per_problem"]),
        key=lambda row: row["problem_index"],
    )
    indices = [row["problem_index"] for row in per_problem]
    if indices != list(range(len(per_problem))):
        raise ValueError("shards contain duplicate or missing problem indices")

    n_correct = sum(row["n_correct"] for row in per_problem)
    n_samples = sum(row["n"] for row in per_problem)
    k = reference["k"]
    result = {
        **reference,
        "shard": None,
        "n_problems": len(per_problem),
        "pass@1": n_correct / n_samples,
        f"pass@{k}": sum(row["n_correct"] > 0 for row in per_problem) / len(per_problem),
        "per_problem": per_problem,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps({key: value for key, value in result.items() if key != "per_problem"}, indent=2)
    )


if __name__ == "__main__":
    main()

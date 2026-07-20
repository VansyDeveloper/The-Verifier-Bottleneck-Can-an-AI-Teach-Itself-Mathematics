"""pass@k eval on the SFT training prompts: the covered pool prompts
(seed=777 train problems that have >=1 correct completion in pool_offline.jsonl).
Output format matches eval_passk.py (per_problem list) for gen_slide_charts.py.
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

from modcomp.checker import is_correct
from modcomp.gen import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="pool_offline.jsonl")
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--eval-seed", type=int, default=123)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--prime-max", type=int, default=29)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--style", default="compact")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    covered = {json.loads(line)["prompt"] for line in open(args.pool)}
    rows = make_dataset(
        args.n_prompts,
        seed=args.seed,
        split="train",
        k_min=args.k_min,
        k_max=args.k_max,
        prime_max=args.prime_max,
        style=args.style,
    )
    rows = [r for r in rows if r["prompt"] in covered]
    rows = rows[args.shard_id :: args.num_shards]
    print(f"{len(rows)} covered prompts (shard {args.shard_id}/{args.num_shards})")
    llm = LLM(model=args.model, gpu_memory_utilization=0.85)
    sp = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.eval_seed,
        stop=["\nProblem:", "\nSolved:", "\n\n\n"],
    )
    outs = llm.generate([r["prompt"] for r in rows], sp)

    per_problem = []
    for r, o in zip(rows, outs):
        c = sum(is_correct(cand.text, r["p"], r["target_A"], r["target_B"]) for cand in o.outputs)
        per_problem.append({"n_correct": c, "n": len(o.outputs)})
    res = {
        "model": args.model,
        "split": "pool_covered",
        "samples": args.samples,
        "temperature": args.temperature,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "per_problem": per_problem,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

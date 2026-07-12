"""Fixed offline candidate pool for the H1 offline-vs-on-policy test.

Sample the BASE model once on training prompts, keep completions accepted by the
ideal verifier (alpha=1, beta=0), never refresh. sft_offline.py then trains on
this frozen pool; if that raises pass@1 but not pass@64 while on-policy GRPO
raises both, that is direct evidence for the H1 sharpening mechanism.
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

from vllm import LLM, SamplingParams

from modcomp.checker import is_correct
from modcomp.gen import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument(
        "--max-per-prompt",
        type=int,
        default=4,
        help="cap of accepted completions kept per prompt (dedup pressure)",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--prime-max", type=int, default=29)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--style", default="compact")
    ap.add_argument("--out", default="pool_offline.jsonl")
    args = ap.parse_args()

    rows = make_dataset(
        args.n_prompts,
        seed=args.seed,
        split="train",
        k_min=args.k_min,
        k_max=args.k_max,
        prime_max=args.prime_max,
        style=args.style,
    )
    llm = LLM(model=args.model, gpu_memory_utilization=0.85)
    sp = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["\nProblem:", "\nSolved:", "\n\n\n"],
    )
    outs = llm.generate([r["prompt"] for r in rows], sp)

    kept, covered, n_correct = 0, 0, 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            good, seen = [], set()
            for s in o.outputs:
                ok = is_correct(s.text, r["p"], r["target_A"], r["target_B"])
                n_correct += ok
                if ok and s.text not in seen:
                    seen.add(s.text)
                    good.append(s.text)
            covered += bool(good)
            for text in good[: args.max_per_prompt]:
                f.write(json.dumps({"prompt": r["prompt"], "completion": text}) + "\n")
                kept += 1
    stats = dict(
        n_prompts=len(rows),
        samples=args.samples,
        pool_pass1=n_correct / (len(rows) * args.samples),
        coverage=covered / len(rows),
        kept=kept,
    )
    json.dump(stats, open(args.out + ".stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

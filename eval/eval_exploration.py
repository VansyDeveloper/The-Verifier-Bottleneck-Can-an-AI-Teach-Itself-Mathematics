"""Measure effective-support exploration at fixed candidate count K.

The reference policy pi is the model sampled at --reference-temperature. Each
temperature or mixture defines rho. The reported epsilon values are empirical
mass outside pi's effective support at resolution eta=1/resolution_m.

Example (single GPU):
  python eval/eval_exploration.py --model Qwen/Qwen3-0.6B-Base --out epsilon.json
"""

import argparse
import json
import os
import random
from math import sqrt
from statistics import fmean, stdev

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from modcomp.exploration import novelty_metrics, outcome_class
from modcomp.gen import make_dataset


METRICS = (
    "epsilon_raw",
    "epsilon_parseable",
    "epsilon_correct",
    "parse_rate",
    "pass@1",
    "pass@k",
    "effective_support_size",
    "novel_parseable_classes",
)


def aggregate(rows):
    out = {}
    for key in METRICS:
        values = [row[key] for row in rows if row[key] is not None]
        if values:
            out[key] = fmean(values)
            out[f"se_{key}"] = stdev(values) / sqrt(len(values)) if len(values) > 1 else None
    novel_parseable = sum(row["novel_parseable_count"] for row in rows)
    novel = sum(row["novel_count"] for row in rows)
    novel_correct = sum(row["novel_correct_count"] for row in rows)
    out["novel_precision"] = novel_correct / novel if novel else None
    out["novel_parseable_precision"] = (
        novel_correct / novel_parseable if novel_parseable else None
    )
    return out


def mix_candidate_rows(base, proposal, weight, seed):
    """Make K iid draws from (1-weight)*base + weight*proposal per problem."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must lie in [0, 1]")
    if len(base) != len(proposal):
        raise ValueError("base and proposal must contain the same problems")
    mixed = []
    proposal_count = 0
    total_count = 0
    for row_index, (base_row, proposal_row) in enumerate(zip(base, proposal)):
        if len(base_row) != len(proposal_row):
            raise ValueError("base and proposal rows must have equal length")
        rng = random.Random(seed + row_index)
        row = []
        for base_item, proposal_item in zip(base_row, proposal_row):
            use_proposal = rng.random() < weight
            row.append(proposal_item if use_proposal else base_item)
            proposal_count += int(use_proposal)
            total_count += 1
        mixed.append(row)
    realized_weight = proposal_count / total_count if total_count else 0.0
    return mixed, realized_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--k", type=int, default=32, help="fixed candidates per intervention")
    ap.add_argument(
        "--reference-samples",
        type=int,
        default=None,
        help="independent samples used to estimate pi (default: 4*k)",
    )
    ap.add_argument(
        "--resolution-m",
        type=int,
        nargs="+",
        default=None,
        help="one or more eta=1/m resolutions, reusing the same samples (default: m=k)",
    )
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.3, 0.7, 1.0, 1.3])
    ap.add_argument("--reference-temperature", type=float, default=1.0)
    ap.add_argument("--proposal-temperature", type=float, default=1.3)
    ap.add_argument("--mixture-weights", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--prime-max", type=int, default=29)
    ap.add_argument("--style", choices=["verbose", "compact"], default="compact")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument(
        "--backend",
        choices=["auto", "vllm", "transformers"],
        default="auto",
        help="auto uses vLLM on CUDA and Transformers elsewhere",
    )
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
        help="Transformers-only generation chunk; does not change candidate K",
    )
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--out", default="results/data/exploration_sweep.json")
    args = ap.parse_args()

    if args.k <= 0 or args.n_problems <= 0 or args.generation_batch_size <= 0:
        ap.error("--k, --n-problems, and --generation-batch-size must be positive")
    reference_samples = 4 * args.k if args.reference_samples is None else args.reference_samples
    resolution_ms = list(
        dict.fromkeys([args.k] if args.resolution_m is None else args.resolution_m)
    )
    if reference_samples <= 0 or any(resolution_m <= 0 for resolution_m in resolution_ms):
        ap.error("--reference-samples and every --resolution-m must be positive")
    if any(not 0 <= weight <= 1 for weight in args.mixture_weights):
        ap.error("--mixture-weights must lie in [0, 1]")
    all_temperatures = [
        *args.temperatures,
        args.reference_temperature,
        args.proposal_temperature,
    ]
    if any(temperature <= 0 for temperature in all_temperatures):
        ap.error("temperatures must be positive when drawing more than one candidate")

    rows = make_dataset(
        args.n_problems,
        seed=args.seed + 10_000,
        split=args.split,
        k_min=args.k_min,
        k_max=args.k_max,
        prime_max=args.prime_max,
        style=args.style,
    )
    prompts = [row["prompt"] for row in rows]
    import torch

    backend = args.backend
    if backend == "auto":
        backend = "vllm" if torch.cuda.is_available() else "transformers"

    resolved_device = None
    resolved_dtype = None
    if backend == "vllm":
        if args.device not in {"auto", "cuda"}:
            ap.error("the vllm backend only supports --device auto/cuda in this repository")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        resolved_device = "cuda"
        resolved_dtype = "vllm-auto"

        def sample(temperature, n, seed):
            params = [
                SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=1.0,
                    max_tokens=args.max_tokens,
                    seed=seed + problem_index,
                    stop=["\nProblem:", "\nSolved:", "\n\n\n"],
                )
                for problem_index in range(len(prompts))
            ]
            outputs = llm.generate(prompts, params)
            candidates = [
                [outcome_class(candidate.text, row["p"]) for candidate in output.outputs]
                for row, output in zip(rows, outputs)
            ]
            if any(len(row) != n for row in candidates):
                raise RuntimeError("vLLM returned the wrong number of candidates")
            return candidates

    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.device == "auto":
            if torch.backends.mps.is_available():
                resolved_device = "mps"
            elif torch.cuda.is_available():
                resolved_device = "cuda"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = args.device
        if resolved_device == "mps" and not torch.backends.mps.is_available():
            ap.error("MPS is unavailable; check Apple Silicon PyTorch installation")
        if resolved_device == "cuda" and not torch.cuda.is_available():
            ap.error("CUDA is unavailable")
        dtype = torch.float16 if resolved_device == "mps" else (
            torch.bfloat16 if resolved_device == "cuda" else torch.float32
        )
        resolved_dtype = str(dtype).removeprefix("torch.")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model.to(resolved_device)
        model.eval()
        stop_strings = ("\nProblem:", "\nSolved:", "\n\n\n")

        def trim_stop(text):
            positions = [text.find(stop) for stop in stop_strings if stop in text]
            return text[: min(positions)] if positions else text

        def sample(temperature, n, seed):
            candidates = []
            for problem_index, (prompt, row) in enumerate(zip(prompts, rows)):
                encoded = tokenizer(prompt, return_tensors="pt")
                encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
                prompt_length = encoded["input_ids"].shape[1]
                texts = []
                for chunk_index, start in enumerate(range(0, n, args.generation_batch_size)):
                    chunk_size = min(args.generation_batch_size, n - start)
                    torch.manual_seed(seed + problem_index + 1_000_000 * chunk_index)
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            do_sample=True,
                            temperature=temperature,
                            top_p=1.0,
                            max_new_tokens=args.max_tokens,
                            num_return_sequences=chunk_size,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    texts.extend(
                        trim_stop(tokenizer.decode(tokens[prompt_length:], skip_special_tokens=True))
                        for tokens in generated
                    )
                if len(texts) != n:
                    raise RuntimeError("Transformers returned the wrong number of candidates")
                candidates.append([outcome_class(text, row["p"]) for text in texts])
            return candidates

    reference = sample(args.reference_temperature, reference_samples, args.seed + 100_000)
    if any(len(row) != reference_samples for row in reference):
        raise RuntimeError("reference sample count does not match M")
    reported_temperatures = list(dict.fromkeys([*args.temperatures, args.reference_temperature]))
    temperatures = list(
        dict.fromkeys([*reported_temperatures, args.proposal_temperature])
    )
    sampled = {
        temperature: sample(temperature, args.k, args.seed + 200_000 + 10_000 * index)
        for index, temperature in enumerate(temperatures)
    }
    if any(len(row) != args.k for candidates in sampled.values() for row in candidates):
        raise RuntimeError("candidate sample count does not match K")

    def evaluate(name, candidates, metadata, resolution_m):
        if len(candidates) != len(rows) or any(len(row) != args.k for row in candidates):
            raise RuntimeError("every problem must retain exactly K candidates")
        per_problem = []
        for index, (row, ref, cand) in enumerate(zip(rows, reference, candidates)):
            metrics = novelty_metrics(
                ref,
                cand,
                (row["target_A"], row["target_B"]),
                resolution_m,
            )
            per_problem.append(
                {"problem_index": index, "p": row["p"], "k_funcs": row["k"], **metrics}
            )
        return {
            "name": f"m{resolution_m}_{name}",
            "resolution_m": resolution_m,
            "eta": 1 / resolution_m,
            "candidate_k": args.k,
            **metadata,
            "mean": aggregate(per_problem),
            "per_problem": per_problem,
        }

    conditions = [
        evaluate(
            f"temperature_{temperature:g}",
            sampled[temperature],
            {"kind": "temperature", "temperature": temperature},
            resolution_m,
        )
        for resolution_m in resolution_ms
        for temperature in reported_temperatures
    ]

    base = sampled[args.reference_temperature]
    proposal = sampled[args.proposal_temperature]
    for weight_index, weight in enumerate(args.mixture_weights):
        mixed, realized_weight = mix_candidate_rows(
            base,
            proposal,
            weight,
            args.seed + 300_000 + 10_000 * weight_index,
        )
        for resolution_m in resolution_ms:
            conditions.append(
                evaluate(
                    f"mixture_{weight:g}",
                    mixed,
                    {
                        "kind": "mixture",
                        "requested_weight": weight,
                        "realized_weight": realized_weight,
                        "base_temperature": args.reference_temperature,
                        "proposal_temperature": args.proposal_temperature,
                    },
                    resolution_m,
                )
            )

    baselines = {
        resolution_m: next(
            condition
            for condition in conditions
            if condition["resolution_m"] == resolution_m
            and condition["kind"] == "temperature"
            and condition["temperature"] == args.reference_temperature
        )
        for resolution_m in resolution_ms
    }
    for condition in conditions:
        baseline = baselines[condition["resolution_m"]]
        for metric in ("epsilon_raw", "epsilon_parseable", "epsilon_correct"):
            deltas = [
                row[metric] - base_row[metric]
                for row, base_row in zip(condition["per_problem"], baseline["per_problem"])
            ]
            condition["mean"][f"delta_{metric}"] = fmean(deltas)
            condition["mean"][f"se_delta_{metric}"] = (
                stdev(deltas) / sqrt(len(deltas)) if len(deltas) > 1 else None
            )

    result = {
        "definition": {
            "phi": "canonical parsed answer (A mod p, B mod p); raw also treats null as bottom",
            "eta": [1 / resolution_m for resolution_m in resolution_ms],
            "resolution_m": resolution_ms,
            "reference_samples": reference_samples,
            "candidate_k": args.k,
            "estimator": "epsilon_hat outside plug-in support from an independent finite reference sample",
            "warning": (
                "support_hat is random and threshold-biased; repeat seeds and vary reference "
                "size/resolution before claims; raw includes unparseable bottom; answer-level "
                "novelty is not evidence of a new reasoning strategy"
            ),
        },
        "config": {
            "model": args.model,
            "split": args.split,
            "seed": args.seed,
            "n_problems": args.n_problems,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "prime_max": args.prime_max,
            "style": args.style,
            "max_tokens": args.max_tokens,
            "reference_temperature": args.reference_temperature,
            "proposal_temperature": args.proposal_temperature,
            "temperatures": args.temperatures,
            "mixture_weights": args.mixture_weights,
            "resolution_m": resolution_ms,
            "backend": backend,
            "device": resolved_device,
            "dtype": resolved_dtype,
            "generation_batch_size": (
                args.generation_batch_size if backend == "transformers" else None
            ),
        },
        "conditions": conditions,
    }
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {args.out}")
    print(
        f"{'condition':>18} {'eps_parse':>9} {'d_parse':>8} {'eps_correct':>11} "
        f"{'d_correct':>10} {'parse':>8} {'pass@k':>8}"
    )
    for condition in conditions:
        mean = condition["mean"]
        print(
            f"{condition['name']:>18} {mean['epsilon_parseable']:>9.3f} "
            f"{mean['delta_epsilon_parseable']:>8.3f} {mean['epsilon_correct']:>11.3f} "
            f"{mean['delta_epsilon_correct']:>10.3f} {mean['parse_rate']:>8.3f} "
            f"{mean['pass@k']:>8.3f}"
        )


if __name__ == "__main__":
    main()

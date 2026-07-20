"""Pull representative model reasoning traces from the logged rollout parquets.

Writes results/traces/checker_examples.md with, per selected run:
  - a late-training CORRECT solve (model reasons through the composition)
  - a WRONG-but-accepted rollout where informative<->noisy checker matters
  - an early vs late pair to show the reasoning tightening up

  python analysis/extract_traces.py --runs-dir runs
"""

import argparse
import glob
import os

import pandas as pd


def load_run(rd):
    pqs = sorted(glob.glob(os.path.join(rd, "completions", "*.parquet")))
    if not pqs:
        return None
    return pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)


def problem_only(prompt):
    # keep just the final (unsolved) problem block, drop the few-shot preamble
    idx = prompt.rfind("Problem:")
    return prompt[idx:].strip() if idx >= 0 else prompt.strip()


def block(title, row):
    return (
        f"### {title}\n\n"
        f"*step {int(row.step)} · checker_reward={row.reward_noisy_checker:.0f} · "
        f"true_correct={row.true_accuracy:.0f}*\n\n"
        f"```\n{problem_only(row.prompt)}\n"
        f"{row.completion.strip()}\n```\n\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results/traces")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    md = [
        "# Model reasoning traces\n",
        "Rollouts logged during GRPO (`log_completions=True`). The prompt shows "
        "two solved few-shot examples then the problem to solve; only the final "
        "problem block is reproduced here. Reward is the noisy checker's verdict; "
        "`true_correct` is the noiseless ground truth.\n",
    ]

    # perfect-judge run: clean correct solves, early vs late
    a10 = load_run(os.path.join(args.runs_dir, "a1.0_b0.0_p29_t1.0_s0"))
    if a10 is not None:
        md.append("## Perfect checker (α=1, β=0) — reasoning tightens over training\n")
        early = a10[(a10.step <= 5) & (a10.true_accuracy == 1)]
        late = a10[(a10.step >= 240) & (a10.true_accuracy == 1)]
        if len(early):
            md.append(block("Early correct solve", early.iloc[0]))
        if len(late):
            for i in range(min(2, len(late))):
                md.append(block(f"Late correct solve #{i + 1}", late.iloc[i * 3 % len(late)]))
        # a late still-wrong one, if any, to show residual failure mode
        latewrong = a10[(a10.step >= 240) & (a10.true_accuracy == 0)]
        if len(latewrong):
            md.append(block("Late incorrect (residual failure mode)", latewrong.iloc[0]))

    # collapse run: verifier accepts a WRONG answer (reward=1, true=0)
    a05 = load_run(os.path.join(args.runs_dir, "a0.5_b0.5_p29_t1.0_s0"))
    if a05 is not None:
        md.append("## Break-even checker (α=β=0.5) — reward decoupled from truth\n")
        hack = a05[(a05.reward_noisy_checker == 1) & (a05.true_accuracy == 0)]
        if len(hack):
            md.append(block("WRONG answer the checker still rewarded", hack.iloc[len(hack) // 2]))
        miss = a05[(a05.reward_noisy_checker == 0) & (a05.true_accuracy == 1)]
        if len(miss):
            md.append(block("CORRECT answer the checker rejected", miss.iloc[len(miss) // 2]))

    # hard modulus solved via curriculum
    p97 = load_run(os.path.join(args.runs_dir, "curric_p97_s0"))
    if p97 is not None:
        md.append("## Curriculum top rung (p≤97, perfect judge) — hardest modulus solved\n")
        hard = p97[(p97.step >= 140) & (p97.true_accuracy == 1)]
        # prefer an actual large-prime instance
        big = hard[hard.prompt.str.contains("p = 97") | hard.prompt.str.contains("p = 89")]
        pick = big if len(big) else hard
        for i in range(min(2, len(pick))):
            md.append(block(f"Solve #{i + 1}", pick.iloc[i]))

    with open(os.path.join(args.out, "checker_examples.md"), "w") as f:
        f.write("\n".join(md))
    print("wrote", os.path.join(args.out, "checker_examples.md"), "-", len(md), "blocks")


if __name__ == "__main__":
    main()

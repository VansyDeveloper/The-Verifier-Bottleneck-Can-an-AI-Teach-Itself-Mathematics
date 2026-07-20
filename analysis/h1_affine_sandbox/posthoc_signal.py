"""Post-hoc signal-density metrics from GRPO completion parquets.

For each run and each logged step:
  a_t        = mean(true_accuracy)                    (policy accuracy on train batch)
  q          = mean(reward_noisy_checker)             (acceptance rate)
  frac_inf   = fraction of prompt-groups with reward std > 0 (informative groups)
  precision  = P(correct | accepted)
  recall_sig = P(accepted | correct)  (~alpha check)
Aggregated into a local JSON per run: results_remote/posthoc_signal.json
"""

import glob
import json
import os
import pandas as pd

RUNS = [
    "ph3n25p29_od1_a0.6_b0.0",
    "ph3n25p29_od2_a1.0_b0.4",
    "ph3n25p29_mid_a1.0_b0.0",
    "ph3n25p29_mid_a0.8_b0.2",
    "ph3n25p29_mid_a0.6_b0.4",
    "ph3n25p29_ctlh_a0.5_b0.5",
    "ph3n25p29_ctlf_a0.5_b0.5",
]

out = {}
for run in RUNS:
    files = sorted(glob.glob(f"runs/{run}/completions/completions_*.parquet"))
    if not files:
        print("skip", run)
        continue
    rows = []
    for f in files:
        df = pd.read_parquet(f)
        step = int(df["step"].iloc[0])
        acc = df["true_accuracy"].values.astype(float)
        rew = df["reward_noisy_checker"].values.astype(float)
        a_t = acc.mean()
        q = rew.mean()
        g = df.groupby("prompt")["reward_noisy_checker"].std(ddof=0)
        frac_inf = float((g > 0).mean())
        n_acc = rew.sum()
        precision = float((acc * rew).sum() / n_acc) if n_acc > 0 else None
        n_cor = acc.sum()
        recall = float((acc * rew).sum() / n_cor) if n_cor > 0 else None
        rows.append(
            dict(
                step=step,
                a=float(a_t),
                q=float(q),
                frac_informative=frac_inf,
                precision=precision,
                recall=recall,
                n=len(df),
            )
        )
    rows.sort(key=lambda r: r["step"])
    out[run] = rows
    last = rows[-1]
    mid = rows[len(rows) // 2]
    print(
        f"{run}: steps={len(rows)}  a_final={last['a']:.3f}  "
        f"q_mid={mid['q']:.3f}  inf_mid={mid['frac_informative']:.3f}  "
        f"prec_mid={mid['precision'] if mid['precision'] is None else round(mid['precision'], 3)}"
    )

os.makedirs("results", exist_ok=True)
json.dump(out, open("results_remote/posthoc_signal.json", "w"))
print("saved results_remote/posthoc_signal.json")

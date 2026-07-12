"""Post-hoc exploration & format metrics from GRPO completion parquets (CPU only).

Per run, per step:
  uniq_ans       = mean over prompt-groups of (#unique completions / group size)
  frac_degenerate= fraction of groups where all completions identical
  pos_coverage   = fraction of groups with >=1 truly-correct completion
  frac_unreduced = among correct completions, share whose A or B is outside [0, p)
                   (the checker-loophole magnitude; needs 'p' recoverable from prompt)
Aggregates every run under runs/ph3n25p29_* into results_remote/posthoc_explore.json.
"""

import glob
import json
import os
import re
import pandas as pd

from modcomp.checker import extract_answer as last_ab  # first "Answer: A=.., B=.."


def prompt_p(prompt):
    m = re.findall(r"^p = (\d+)$", prompt, re.M)
    return int(m[-1]) if m else None


out = {}
for rd in sorted(glob.glob("runs/ph3n25p29_*")):
    run = os.path.basename(rd)
    files = sorted(glob.glob(f"{rd}/completions/completions_0*.parquet"))
    if not files:
        continue
    rows = []
    for f in files:
        df = pd.read_parquet(f)
        step = int(df["step"].iloc[0])
        g = df.groupby("prompt")
        uniq = g["completion"].nunique() / g.size()
        cov = g["true_accuracy"].max()
        unred_n = unred_d = 0
        for _, r in df[df["true_accuracy"] > 0].iterrows():
            ab = last_ab(r["completion"])
            p = prompt_p(r["prompt"])
            if ab and p:
                unred_d += 1
                if not (0 <= ab[0] < p and 0 <= ab[1] < p):
                    unred_n += 1
        rows.append(
            dict(
                step=step,
                uniq_ans=float(uniq.mean()),
                frac_degenerate=float((uniq * g.size() <= 1).mean()),
                pos_coverage=float((cov > 0).mean()),
                frac_unreduced=(unred_n / unred_d) if unred_d else None,
                n_correct=int(df["true_accuracy"].sum()),
            )
        )
    rows.sort(key=lambda r: r["step"])
    out[run] = rows
    a, z = rows[0], rows[-1]
    print(
        f"{run}: uniq {a['uniq_ans']:.2f}->{z['uniq_ans']:.2f}  "
        f"degen {a['frac_degenerate']:.2f}->{z['frac_degenerate']:.2f}  "
        f"cov {a['pos_coverage']:.2f}->{z['pos_coverage']:.2f}  "
        f"unred {a['frac_unreduced']}->{z['frac_unreduced']}"
    )

os.makedirs("results", exist_ok=True)
json.dump(out, open("results_remote/posthoc_explore.json", "w"))
print("saved results_remote/posthoc_explore.json")

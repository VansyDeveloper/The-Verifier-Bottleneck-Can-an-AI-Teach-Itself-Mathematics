"""Bootstrap CIs over problems for pass@k and Δpass@k vs base.

For each result JSON: unbiased per-problem pass@k = 1 - C(n-c, k)/C(n, k).
CI: percentile bootstrap (10k resamples) over the 200 problems.
Base and trained evals use the same eval seed => identical problem lists
(verified per file by (p, k_funcs) sequence). Δ is therefore *paired*:
resample problem indices once, Δ_b = mean_i(rl_i - base_i).
Falls back to independent bootstrap if the problem lists differ.
Outputs results_remote/bootstrap_ci.json and prints a table.
"""

import json
import glob
import os
import numpy as np
from math import comb

RNG = np.random.default_rng(0)
B = 10000


def per_problem_passk(j, k):
    out = []
    for r in j["per_problem"]:
        n, c = r["n"], r["n_correct"]
        kk = min(k, n)
        out.append(1.0 - comb(n - c, kk) / comb(n, kk))
    return np.array(out)


def problem_ids(j):
    return [(r["p"], r["k_funcs"]) for r in j["per_problem"]]


def ci(dist, lo=2.5, hi=97.5):
    return float(np.percentile(dist, lo)), float(np.percentile(dist, hi))


def load(path):
    return json.load(open(path))


results = {}
for d, base_name in [("ph3n25p29", "base_0.6B"), ("ph3easy", "base_0.6B")]:
    bases = {}
    for split in ["eval", "train"]:
        p = f"results_remote/{d}/{base_name}_{split}.json"
        if os.path.exists(p):
            bases[split] = load(p)
    for f in sorted(glob.glob(f"results_remote/{d}/*.json")):
        name = os.path.basename(f)[:-5]
        j = load(f)
        split = j["split"]
        kmax = j["k"]
        entry = {}
        idx = RNG.integers(0, len(j["per_problem"]), size=(B, len(j["per_problem"])))
        for k in [1, kmax]:
            x = per_problem_passk(j, k)
            bm = x[idx].mean(axis=1)
            entry[f"pass@{k}"] = dict(mean=float(x.mean()), ci=ci(bm))
            if split in bases and not name.startswith("base"):
                jb = bases[split]
                xb = per_problem_passk(jb, k)
                if problem_ids(j) == problem_ids(jb):
                    diff = x - xb
                    bd = diff[idx].mean(axis=1)  # paired
                    paired = True
                else:
                    idxb = RNG.integers(0, len(xb), size=(B, len(xb)))
                    bd = bm - xb[idxb].mean(axis=1)
                    paired = False
                entry[f"dpass@{k}"] = dict(
                    mean=float(x.mean() - xb.mean()), ci=ci(bd), paired=paired
                )
        results[f"{d}/{name}"] = entry

json.dump(results, open("results_remote/bootstrap_ci.json", "w"), indent=1)

for name, e in results.items():
    parts = [name.ljust(50)]
    for key in e:
        mean, (low, high) = e[key]["mean"], e[key]["ci"]
        tag = "" if e[key].get("paired", True) else "*"
        parts.append(f"{key}={mean:.3f} [{low:.3f},{high:.3f}]{tag}")
    print("  ".join(parts))

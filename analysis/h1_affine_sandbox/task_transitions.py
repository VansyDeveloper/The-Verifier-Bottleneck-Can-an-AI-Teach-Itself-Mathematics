"""Task-level transitions base -> RL (paired: same 200 problems per split).

A problem is 'solved' for a model if n_correct > 0 within its k samples
(base: k=64; optionally k=256/1024 via --base-suffix). Categories:
retained / forgotten / newly_reached / still_outside.
"""

import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results_remote/ph3n25p29")
ap.add_argument("--base-suffix", default="", help="'' for k64 base, '_k256' etc.")
args = ap.parse_args()


def solved_vec(path):
    j = json.load(open(path))
    return [(r["p"], r["k_funcs"], r["n_correct"] > 0) for r in j["per_problem"]]


out = {}
for split in ["eval", "train"]:
    bp = f"{args.dir}/base_0.6B_{split}{args.base_suffix}.json"
    if not os.path.exists(bp):
        continue
    base = solved_vec(bp)
    for f in sorted(glob.glob(f"{args.dir}/*_{split}.json")):
        name = os.path.basename(f)[:-5]
        if name.startswith("base"):
            continue
        rl = solved_vec(f)
        if [x[:2] for x in rl] != [x[:2] for x in base]:
            print("SKIP unpaired", name)
            continue
        cat = dict(retained=0, forgotten=0, newly_reached=0, still_outside=0)
        for (_, _, b), (_, _, r) in zip(base, rl):
            cat[
                ("retained" if r else "forgotten")
                if b
                else ("newly_reached" if r else "still_outside")
            ] += 1
        out[name] = cat
        print(
            f"{name:50s} retained={cat['retained']:3d} forgotten={cat['forgotten']:3d} "
            f"new={cat['newly_reached']:3d} outside={cat['still_outside']:3d}"
        )

json.dump(
    out, open(f"results_remote/task_transitions{args.base_suffix or '_k64'}.json", "w"), indent=1
)

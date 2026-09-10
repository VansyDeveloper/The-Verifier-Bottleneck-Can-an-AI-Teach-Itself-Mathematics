import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    source = Path("artifacts/reports/three_seed_heldout_statistics.json")
    output = Path("artifacts/reports/heldout_pass_at_32.png")
    data = json.loads(source.read_text(encoding="utf-8"))
    seeds = [str(item["seed"]) for item in data["per_seed"]]
    iid = [100 * item["iid_pass_at_32"] for item in data["per_seed"]]
    prefix = [100 * item["prefix_pass_at_32"] for item in data["per_seed"]]
    x = np.arange(len(seeds))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.bar(x - width / 2, iid, width, label="IID", color="#667085")
    axis.bar(x + width / 2, prefix, width, label="Prefix-balanced", color="#1570EF")
    axis.set_ylabel("Held-out PLAN depth-3 pass@32 (%)")
    axis.set_xlabel("Seed")
    axis.set_xticks(x, seeds)
    axis.set_ylim(0, max(prefix) + 8)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()

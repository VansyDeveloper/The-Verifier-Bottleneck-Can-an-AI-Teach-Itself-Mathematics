"""Depth-4 evaluation splits for exhaustive ranking.

The registered exhaustive grid stops at depth 3 (125 programs). Depth 2 is
uninterpretable at Hit@32 because all 25 programs fit inside K, and depth 3 is
the only informative cell. That leaves the depth axis resting on a single point.
Depth 4 has 5^4 = 625 candidates, so Hit@32 is far from saturated, and it is the
one place where the external package reports a much weaker effect (+5.20 pp
against their +27.40 pp at depth 3) - a claim this project currently cannot
check either way.

Cost is the prefix tree: 1 + 5 + 25 + 125 = 156 scored prefixes per task against
31 at depth 3, so a 125-task split is roughly 37 minutes per adapter on the
RTX 3050 Ti.

Every generated task is rejected if its exact key already appears in any
pre-existing split, including the depth-3 capacity and motif families.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_capacity_data import (  # noqa: E402
    DATA,
    EXISTING,
    HELDOUT_PRIMES,
    TRAIN_PRIMES,
    build,
    exact_key,
)
from vbexp.experiment import write_json  # noqa: E402
from vbexp.io import read_tasks  # noqa: E402

# Splits created after prepare_capacity_data.py was written; they must also be
# excluded, otherwise a depth-4 task could collide with a depth-4 key elsewhere.
ALSO_EXISTING = [
    "sft_train_composition",
    "sft_train_composition_nomotif",
    "capacity_d2_train",
    "capacity_d3_train",
    "capacity_d2_heldout",
    "capacity_d3_heldout",
    "motif_a_d3",
    "motif_b_d3",
    "motif_c_d3",
    "motif_d_d3",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=125)
    parser.add_argument("--heldout", action="store_true", help="also build the 11/17 split")
    args = parser.parse_args()

    taken: set = set()
    for name in [*EXISTING, *ALSO_EXISTING]:
        path = DATA / f"{name}.jsonl"
        if path.exists():
            taken.update(exact_key(task) for task in read_tasks(path))

    records = [build("capacity_d4_train", TRAIN_PRIMES, (4,), args.count, 40011, taken)]
    if args.heldout:
        records.append(build("capacity_d4_heldout", HELDOUT_PRIMES, (4,), args.count, 40012, taken))

    write_json(DATA / "capacity_d4_manifest.json", {"files": records})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

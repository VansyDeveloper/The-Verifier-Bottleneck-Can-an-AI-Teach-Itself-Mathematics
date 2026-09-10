from __future__ import annotations

import json
from pathlib import Path

from vbexp.experiment import sha256_file
from vbexp.generator import GenerationSpec, generate_many
from vbexp.io import write_tasks


def main() -> None:
    spec = GenerationSpec(
        split="sft_calibration_sh1",
        mode="apply",
        primes=(5, 7, 13, 19, 23, 29),
        degree_caps=(2, 3, 4),
        depths=(1,),
        operations=("SH1",),
        require_order_sensitive=False,
    )
    path = Path("artifacts/data/pilot/sft_calibration_sh1.jsonl")
    write_tasks(path, generate_many(spec, 4000, seed=106))
    print(json.dumps({"path": str(path), "count": 4000, "sha256": sha256_file(path)}, indent=2))


if __name__ == "__main__":
    main()

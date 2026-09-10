from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbexp.experiment import sha256_file, write_json
from vbexp.generator import GenerationSpec, generate_many
from vbexp.io import write_tasks


def write_split(name: str, mode: str, depth: tuple[int, ...], count: int, seed: int, primes=(5, 7)) -> Path:
    spec = GenerationSpec(
        split=name,
        mode=mode,
        primes=tuple(primes),
        degree_caps=(2, 3),
        depths=depth,
        require_order_sensitive=any(value > 1 for value in depth),
    )
    path = Path("artifacts/data/smoke") / f"{name}.jsonl"
    write_tasks(path, generate_many(spec, count, seed))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["smoke", "pilot"], default="smoke")
    args = parser.parse_args()
    if args.tier == "smoke":
        train_each, validation_each, dev_count, final_count = 10, 10, 20, 20
    else:
        train_each, validation_each, dev_count, final_count = 1000, 100, 100, 200
    root = Path("artifacts/data") / args.tier

    def split(name: str, mode: str, depths: tuple[int, ...], count: int, seed: int, primes=(5, 7)):
        spec = GenerationSpec(
            split=name,
            mode=mode,
            primes=tuple(primes),
            degree_caps=(2, 3),
            depths=depths,
            require_order_sensitive=any(value > 1 for value in depths),
        )
        path = root / f"{name}.jsonl"
        write_tasks(path, generate_many(spec, count, seed))
        return path

    paths = [
        split("sft_train_apply", "apply", (1,), train_each, 100),
        split("sft_train_plan", "plan", (1,), train_each, 101),
        split("sft_validation_apply", "apply", (1,), validation_each, 102),
        split("sft_validation_plan", "plan", (1,), validation_each, 103),
        split("dev_exploration", "plan", (2, 3), dev_count, 104),
        split("final_like_heldout", "plan", (3,), final_count, 105, primes=(11, 17)),
    ]
    records = [
        {"path": path.as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]
    write_json(root / "split_manifest.json", {"tier": args.tier, "files": records})
    frozen = next(item for item in records if item["path"].endswith("final_like_heldout.jsonl"))
    prereg = Path(f"artifacts/preregistration_{args.tier}.md")
    prereg.write_text(
        f"# {args.tier.title()} preregistration\n\n"
        f"This tier is {'technical only' if args.tier == 'smoke' else 'preliminary only'} "
        "and supports no confirmatory scientific conclusion.\n\n"
        f"- Frozen final-like file: `{frozen['path']}`\n"
        f"- SHA256: `{frozen['sha256']}`\n"
        f"- Seed: 0\n- Candidate budget K: {4 if args.tier == 'smoke' else 16}\n"
        f"- GRPO group size G: {4 if args.tier == 'smoke' else 8}\n"
        f"- GRPO steps: at most {5 if args.tier == 'smoke' else 150}\n"
        "- Primary scientific metric remains unchanged and is not evaluated by this smoke run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"files": records, "preregistration": str(prereg)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

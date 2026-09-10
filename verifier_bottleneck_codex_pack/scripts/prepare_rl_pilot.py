from pathlib import Path

from vbexp.experiment import sha256_file
from vbexp.generator import GenerationSpec, generate_many
from vbexp.io import write_tasks


def main():
    spec = GenerationSpec(
        split="rl_train_pilot",
        mode="plan",
        primes=(5, 7, 13, 19, 23, 29),
        degree_caps=(2, 3, 4),
        depths=(2, 3),
        require_order_sensitive=True,
    )
    path = Path("artifacts/data/pilot/rl_train.jsonl")
    write_tasks(path, generate_many(spec, 500, seed=107))
    print(f"{path} sha256={sha256_file(path)} count=500")


if __name__ == "__main__":
    main()

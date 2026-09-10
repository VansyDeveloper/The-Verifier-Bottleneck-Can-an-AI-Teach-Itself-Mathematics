from pathlib import Path

from vbexp.generator import GenerationSpec, generate_many
from vbexp.io import write_tasks


def main():
    output = Path("artifacts/data/smoke")
    output.mkdir(parents=True, exist_ok=True)
    apply_spec = GenerationSpec("smoke", "apply", (5,7), (2,3), (1,2), require_order_sensitive=False)
    plan_spec = GenerationSpec("smoke", "plan", (5,7), (2,3), (2,), require_order_sensitive=True)
    write_tasks(output / "apply.jsonl", generate_many(apply_spec, 20, 0))
    write_tasks(output / "plan.jsonl", generate_many(plan_spec, 20, 1))
    print(output)


if __name__ == "__main__":
    main()

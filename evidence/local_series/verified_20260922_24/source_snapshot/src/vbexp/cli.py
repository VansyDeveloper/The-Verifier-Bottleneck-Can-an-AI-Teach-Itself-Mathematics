from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import GenerationSpec, generate_many
from .io import read_tasks, write_tasks
from .search import all_solutions, shortest_solutions


def cmd_generate(args: argparse.Namespace) -> None:
    spec = GenerationSpec(
        split=args.split,
        mode=args.mode,
        primes=tuple(args.primes),
        degree_caps=tuple(args.degree_caps),
        depths=tuple(args.depths),
        require_order_sensitive=not args.allow_order_insensitive,
        require_unique_shortest=args.unique_shortest,
        min_shortest_solutions=args.min_shortest_solutions,
    )
    tasks = generate_many(spec, count=args.count, seed=args.seed)
    write_tasks(args.output, tasks)
    print(json.dumps({"output": str(args.output), "count": len(tasks)}, ensure_ascii=False))


def cmd_enumerate(args: argparse.Namespace) -> None:
    tasks = read_tasks(args.input)
    for task in tasks[: args.limit]:
        if task.target is None or task.max_steps is None:
            continue
        shortest = shortest_solutions(task.start, task.target, task.p, task.operations, task.max_steps)
        all_valid = all_solutions(task.start, task.target, task.p, task.operations, task.max_steps)
        print(json.dumps({
            "task_id": task.task_id,
            "shortest": shortest,
            "all_count": len(all_valid),
        }, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vbexp")
    sub = parser.add_subparsers(required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--split", default="smoke")
    generate.add_argument("--mode", choices=["apply", "plan"], required=True)
    generate.add_argument("--primes", type=int, nargs="+", default=[5, 7])
    generate.add_argument("--degree-caps", type=int, nargs="+", default=[2, 3])
    generate.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    generate.add_argument("--count", type=int, default=20)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--allow-order-insensitive", action="store_true")
    generate.add_argument("--unique-shortest", action="store_true")
    generate.add_argument("--min-shortest-solutions", type=int, default=1)
    generate.set_defaults(func=cmd_generate)

    enumerate_parser = sub.add_parser("enumerate")
    enumerate_parser.add_argument("--input", type=Path, required=True)
    enumerate_parser.add_argument("--limit", type=int, default=20)
    enumerate_parser.set_defaults(func=cmd_enumerate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

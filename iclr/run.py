"""Prepare Artem's experiment queue; train only with --execute."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path


def recipe_cells(recipes: list[str]) -> list[dict]:
    cells = []
    trace = {"method": "ce", "supervision": "trace", "replay_fraction": 0.2,
             "budget_mode": "examples", "depth3_only": False}
    control = {**trace, "replay_fraction": 1.0, "budget_mode": "target_tokens"}
    for recipe in recipes:
        if recipe == "replay":
            cells.extend({**trace, "replay_fraction": fraction} for fraction in (0., .1, .2, .4))
            cells.append(control)
        elif recipe == "trace":
            cells.extend([trace, {**trace, "supervision": "program"},
                          {**trace, "supervision": "program", "budget_mode": "target_tokens"}])
        elif recipe == "set":
            cells.extend({**trace, "supervision": "program", "depth3_only": True, "replay_fraction": 0.,
                          "method": method} for method in ("ce", "single_norm", "set_mass"))
        elif recipe == "size":
            cells.extend([control, trace])
        else:
            raise ValueError(f"unknown recipe: {recipe}")
    return list({json.dumps(cell, sort_keys=True): cell for cell in cells}.values())


def build_jobs(args: argparse.Namespace) -> list[dict]:
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not all(0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    if args.micro_batch <= 0 or args.effective_batch <= 0 or args.effective_batch % args.micro_batch:
        raise ValueError("effective batch must be a positive multiple of micro batch")
    if (args.epochs <= 0 or (args.num_examples is not None and args.num_examples <= 0)
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("epochs, num-examples and lr must be positive")
    if args.prefix_batch <= 0:
        raise ValueError("prefix-batch must be positive")
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    jobs = []
    for cell in recipe_cells(args.recipe):
        label = (f"{cell['method']}_{cell['supervision']}_r{round(100 * cell['replay_fraction']):02d}"
                 f"_{cell['budget_mode']}_{'d3' if cell['depth3_only'] else 'mixed'}")
        for seed in args.seeds:
            name = f"{model_name}_{label}_seed{seed}"
            jobs.append({**cell, "seed": seed, "model": args.model,
                         "base": str(Path(args.base).resolve()), "data": str(Path(args.data).resolve()),
                         "output": str(Path(args.output).resolve() / "runs" / name),
                         "epochs": args.epochs, "learning_rate": args.learning_rate,
                         "micro_batch": args.micro_batch, "effective_batch": args.effective_batch,
                         "device": args.device, "dtype": "float32", "prefix_batch": args.prefix_batch,
                         "gradient_checkpointing": True,
                         "lora_rank": 32, "lora_alpha": 64, "lora_dropout": .05})
            if args.num_examples is not None:
                jobs[-1]["num_examples"] = args.num_examples
    return jobs


def write_config(path: Path, config: dict | list) -> None:
    """Repeated commands reuse identical configs and reject changed ones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"configuration changed at {path}; choose a new --output")


def comparison_entries(jobs: list[dict], output: Path) -> list[dict]:
    pairs = {}
    for job in jobs:
        if job['method'] != 'ce' or job['supervision'] != 'trace' or job['depth3_only']:
            continue
        arm = {(1., 'target_tokens'): 'atomic_control', (.2, 'examples'): 'composition'}.get(
            (job['replay_fraction'], job['budget_mode']))
        if arm:
            pairs.setdefault(job['seed'], {})[arm] = str(
                (Path(job['output']) / 'eval' / 'metrics.jsonl').relative_to(output))
    return [{'seed': seed, 'arm': arm, 'metrics': paths[arm]}
            for seed, paths in sorted(pairs.items()) if len(paths) == 2
            for arm in ('atomic_control', 'composition')]


def update_comparison(path: Path, entries: list[dict], varying=('replay_fraction', 'budget_mode')) -> None:
    from .common import write_json
    previous = json.loads(path.read_text()) if path.exists() else []
    combined = {}
    context = None
    for entry in [*previous, *entries]:
        key = entry['seed'], entry['arm']
        if key in combined and combined[key] != entry:
            raise ValueError(f'Comparison path changed for {key}')
        combined[key] = entry
        run_name = Path(entry['metrics']).parent.parent.name
        config = json.loads((path.parent / 'configs' / f'{run_name}.json').read_text())
        current = {k: v for k, v in config.items() if k not in ('seed', 'output', *varying)}
        if context is not None and current != context:
            raise ValueError('Paired seeds have different data, base model or training settings; use a new --output')
        context = current
    write_json(path, [combined[key] for key in sorted(combined)])


def ablation_comparisons(jobs: list[dict], output: Path):
    """Pair existing runs; these comparisons add no training jobs."""
    fields = ('method', 'supervision', 'replay_fraction', 'budget_mode', 'depth3_only')
    cells = {}
    for job in jobs:
        cells.setdefault(tuple(job[key] for key in fields), {})[job['seed']] = job
    trace = ('ce', 'trace', .2, 'examples', False)
    specs = [
        ('trace_examples', ('ce', 'program', .2, 'examples', False), trace,
         ('program_only', 'program_trace'), ('supervision',)),
        ('trace_tokens', ('ce', 'program', .2, 'target_tokens', False), trace,
         ('program_token_matched', 'program_trace'), ('supervision', 'budget_mode')),
    ]
    specs.extend((f'replay{round(100 * fraction):02d}',
                  ('ce', 'trace', 0., 'examples', False), ('ce', 'trace', fraction, 'examples', False),
                  ('replay00', f'replay{round(100 * fraction):02d}'), ('replay_fraction',))
                 for fraction in (.1, .2, .4))
    specs.extend((name, (control, 'program', 0., 'examples', True),
                  (treatment, 'program', 0., 'examples', True), (control, treatment), ('method',))
                 for name, control, treatment in (
                     ('set_single_control', 'ce', 'single_norm'),
                     ('set_mass', 'single_norm', 'set_mass'), ('set_vs_ce', 'ce', 'set_mass')))
    for name, control, treatment, arms, varying in specs:
        seeds = sorted(cells.get(control, {}).keys() & cells.get(treatment, {}).keys())
        entries = [{'seed': seed, 'arm': arm, 'metrics': str(
                    (Path(cells[cell][seed]['output']) / 'eval' / 'metrics.jsonl').relative_to(output))}
                   for seed in seeds for arm, cell in zip(arms, (control, treatment))]
        if entries:
            yield name, entries, arms, varying


def completed_jobs(output: Path, reference: dict) -> list[dict]:
    jobs = [json.loads(path.read_text()) for path in sorted((output / 'configs').glob('*.json'))]
    return [job for job in jobs if all(job.get(key) == reference[key] for key in ('model', 'base', 'data'))
            and (Path(job['output']) / 'DONE').is_file()]


def write_results(jobs: list[dict], path: Path, baseline: Path) -> None:
    from .common import file_hash
    base = json.loads((baseline / 'summary.json').read_text())
    rows = []
    for job in jobs:
        directory = Path(job['output'])
        if not (directory / 'DONE').is_file():
            raise ValueError(f'missing completed run: {directory}')
        done = json.loads((directory / 'DONE').read_text())
        if any(done['binding']['config'].get(key) != value for key, value in job.items()):
            raise ValueError(f'Completed run has a different configuration: {directory}')
        if any(done['binding'][key] != base['binding'][key] for key in ('base_hash', 'data_hash')):
            raise ValueError(f'Baseline and completed run have different model/data: {directory}')
        for name in ('budget.json', 'eval/summary.json'):
            if file_hash(directory / name) != done['files'][name]:
                raise ValueError(f'Completed result changed: {directory / name}')
        budget = json.loads((directory / 'budget.json').read_text())
        summary = json.loads((directory / 'eval' / 'summary.json').read_text())
        row = {key: job[key] for key in ('model', 'seed', 'method', 'supervision', 'replay_fraction',
                                        'budget_mode', 'depth3_only', 'epochs', 'learning_rate',
                                        'micro_batch', 'effective_batch')}
        row.update({key: budget.get(key) for key in ('optimizer_steps', 'example_exposures', 'unique_tasks',
                    'all_target_tokens', 'total_forward_tokens', 'wall_seconds', 'peak_memory_bytes')})
        for family, metrics in summary['families'].items():
            row.update({f'{family}_{metric}': value for metric, value in metrics.items()})
        for operation in ('SH1', 'SC2', 'REV', 'AC1', 'AX1'):
            for metric in ('plan', 'apply'):
                accuracy = summary[f'atomic_{metric}']['by_operation'][operation]
                row[f'{operation}_{metric}'] = accuracy
                row[f'{operation}_{metric}_change'] = accuracy - base[f'atomic_{metric}']['by_operation'][operation]
            row[f'{operation}_parse'] = summary['atomic_parse'][operation]['parse_rate']
        rows.append(row)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--recipe", nargs="+", choices=("replay", "trace", "set", "size"), required=True)
    result.add_argument("--data", required=True, help="prepared input bundle")
    result.add_argument("--base", required=True, help="atomic export for this model")
    result.add_argument("--model", required=True, help="model identity, e.g. Qwen/Qwen3-0.6B")
    result.add_argument("--output", required=True)
    result.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    result.add_argument("--num-examples", type=int, help="default: input manifest exposure count")
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--micro-batch", type=int, default=1)
    result.add_argument("--effective-batch", type=int, default=64)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--prefix-batch", type=int, default=8)
    result.add_argument("--execute", action="store_true", help="run the prepared queue sequentially")
    return result


def main(argv: list[str] | None = None) -> None:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    try:
        jobs = build_jobs(args)
        if args.execute:
            for label in ("data", "base"):
                if not Path(getattr(args, label)).is_dir():
                    raise ValueError(f"missing {label} directory: {getattr(args, label)}")
        model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
        output = Path(args.output).resolve()
        baseline = output / f'baseline_{model_name}'
        commands = [[sys.executable, "-m", "iclr.evaluate", "--base", str(Path(args.base).resolve()),
                     "--data", str(Path(args.data).resolve()), "--out",
                     str(baseline),
                     "--device", args.device, "--prefix-batch", str(args.prefix_batch)]]
        for job in jobs:
            name = Path(job["output"]).name
            path = Path(args.output).resolve() / "configs" / f"{name}.json"
            write_config(path, job)
            commands.append([sys.executable, "-m", "iclr.train", "--config", str(path)])
        entries = comparison_entries(jobs, output)
        if entries:
            comparison = output / f'comparison_{model_name}.json'
            update_comparison(comparison, entries)
            print(f'Paired analysis manifest: {comparison}', flush=True)
        analyses = []
        for name, entries, arms, varying in ablation_comparisons(jobs, output):
            comparison = output / f'comparison_{model_name}_{name}.json'
            update_comparison(comparison, entries, varying)
            analyses.append([sys.executable, '-m', 'iclr.analyze', '--manifest', str(comparison),
                             '--arms', *arms, '--output', str(output / 'analysis' / model_name / name), '--plots'])
        for command in commands:
            print(shlex.join(command), flush=True)
        print(f"{len(jobs)} jobs; configs in {Path(args.output).resolve() / 'configs'}", flush=True)
        for command in analyses:
            print('After training: ' + shlex.join(command), flush=True)
        if args.execute:
            for command in commands:
                subprocess.run(command, check=True)
            result_path = output / f'results_{model_name}.csv'
            write_results(completed_jobs(output, jobs[0]), result_path, baseline)
            print(f'Results: {result_path}', flush=True)
    except ValueError as exc:
        arg_parser.error(str(exc))


if __name__ == "__main__":
    main()

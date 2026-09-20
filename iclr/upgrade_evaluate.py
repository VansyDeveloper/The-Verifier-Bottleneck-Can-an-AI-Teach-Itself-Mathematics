"""Prefix-resolved scoring, closed A calibration, and execution on true states."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

import composition_core as core
import composition_eval as legacy
from .calibration import calibrated_rows, family, fit_bias, summarize, write_csv
from .common import code_hash, environment, file_hash, read_jsonl, tree_hash, verify_data, verify_receipt, write_json
from .modeling import PROMPT_VERSION, load, program_scores

SCORE_KEYS = ('score', 'local_score', 'format_score')


def from_prefixes(row, prefixes):
    """Use actual model prefix probabilities, never condition globally normalized leaves."""
    lookup = {tuple(r['prefix']): np.asarray(r['full_action_logprobs'], dtype=np.float64) for r in prefixes}
    expected = {p for d in range(row['depth']) for p in itertools_prefixes(d)}
    if set(lookup) != expected or len(prefixes) != len(expected):
        raise ValueError('Missing or duplicate prefix probabilities')
    ranking = []
    for program in sorted(core.enumerate_programs(row['depth'])):
        full, legal, format_score = 0., 0., 0.
        for i, op in enumerate(program):
            logp = lookup[program[:i]]
            if logp.shape != (5,) or not np.isfinite(logp).all() or np.any(logp > 0):
                raise ValueError('Invalid full-vocabulary action log probabilities')
            logz = float(np.logaddexp.reduce(logp))
            if logz > 1e-6:
                raise ValueError('Legal action probability exceeds one')
            selected = float(logp[core.OPS.index(op)])
            full += selected
            legal += selected - logz
            format_score += logz
        if not np.isclose(full, legal + format_score, atol=1e-10, rtol=0):
            raise ValueError('Full/local/format decomposition failed')
        ranking.append({'program': list(program), 'score': full, 'local_score': legal,
                        'format_score': format_score,
                        'correct': core.verify_program(row['start'], row['target'], program, row['p'])})
    if not np.isclose(sum(np.exp(r['local_score']) for r in ranking), 1, atol=1e-8):
        raise ValueError('Local action policy does not normalize on the complete tree')
    return {**{k: v for k, v in row.items() if k not in ('states', 'correct_programs')}, 'ranking': ranking}


def itertools_prefixes(depth):
    return [()] if depth == 0 else core.enumerate_programs(depth)


def execution_tasks(rows, per_family=50):
    if per_family < 1:
        raise ValueError('Execution panel size must be positive')
    counts, tasks = defaultdict(int), {}
    for row in sorted(rows, key=lambda r: r['task_id']):
        group = family(row), row['depth']
        if row.get('panel_id') or counts[group] >= per_family:
            continue
        counts[group] += 1
        correct = [r['program'] for r in row['ranking'] if r['correct']]
        for program in correct:
            states = core.trajectory(row['start'], program, row['p'])
            specs = [('program', row['start'], program, row['target'])]
            specs.extend(('true_intermediate', list(states[i]), [op], list(states[i + 1])) for i, op in enumerate(program))
            for kind, start, ops, target in specs:
                identity = (row['p'], tuple(start), tuple(ops))
                key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
                if key not in tasks:
                    tasks[key] = {'task_id': key, 'kind': kind, 'p': row['p'], 'start': start,
                                  'program': ops, 'witness': ops, 'depth': len(ops), 'target': target, 'parents': []}
                if row['task_id'] not in tasks[key]['parents']:
                    tasks[key]['parents'].append(row['task_id'])
    return list(tasks.values())


def execute_programs(model, tokenizer, rows, batch_size):
    device = next(model.parameters()).device
    result = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset:offset + batch_size]
            encoded = tokenizer([core.apply_prompt(row) for row in chunk], return_tensors='pt',
                                padding=True, add_special_tokens=False).to(device)
            tokens = model.generate(**encoded, do_sample=False, max_new_tokens=64,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            texts = tokenizer.batch_decode(tokens[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
            for row, text in zip(chunk, texts):
                parsed = None
                try:
                    value = ast.literal_eval(text.strip())
                    if isinstance(value, list) and len(value) == len(row['target']) and all(type(v) is int for v in value):
                        parsed = value
                except (ValueError, SyntaxError, RecursionError):
                    pass
                result.append({**row, 'raw_completion': text, 'parse_ok': parsed is not None,
                               'semantic_correct': parsed is not None and [v % row['p'] for v in parsed] == row['target'],
                               'exact_correct': text.strip() == core.format_state(row['target'])})
    return result


def run(args):
    out = Path(args.out)
    manifest = verify_data(args.data)
    binding = {'base_hash': tree_hash(args.base), 'adapter_hash': tree_hash(args.adapter) if args.adapter else None,
               'data_hash': file_hash(Path(args.data) / 'manifest.json'), 'code_hash': code_hash(),
               'model': args.model, 'seed': args.seed, 'arm': args.arm, 'dtype': args.dtype,
               'prefix_batch': args.prefix_batch, 'device': args.device,
               'execution_tasks_per_family': args.execution_tasks_per_family,
               'reliable_ops': args.reliable_ops, 'protocol': 'full_local_format_prefix_v1',
               'prompt_version': PROMPT_VERSION, 'data_status': manifest['status']}
    for key in ('base_hash', 'adapter_hash'):
        expected = getattr(args, 'expected_' + key)
        if expected is not None and binding[key] != expected:
            raise ValueError(f'Historical checkpoint binding mismatch: {key}')
    if (out / 'DONE').exists():
        if verify_receipt(out)['binding'] != binding:
            raise ValueError('Completed evaluation has different inputs')
        print(f'Already complete: {out}')
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'binding.json').exists() and json.loads((out / 'binding.json').read_text()) != binding:
        raise ValueError('Partial evaluation has different inputs; choose a new output')
    write_json(out / 'binding.json', binding)
    write_json(out / 'environment.json', environment())
    try:
        model, tokenizer, ids = load(args.base, adapter=args.adapter, device=args.device, dtype=args.dtype)
        model.eval()
        # Canonical tokenizer hash excludes mutable padding/truncation state.
        tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
        for key in ('padding', 'truncation'):
            tokenizer_json.pop(key, None)
        write_json(out / 'tokenizer.json', tokenizer_json)
        rows = []
        cal = read_jsonl(Path(args.data) / 'calibration_A.jsonl')
        with (out / 'prefixes.jsonl.partial').open('w') as prefix_file:
            def score(source):
                result = []
                with torch.inference_mode():
                    for row in source:
                        prefixes = []
                        program_scores(model, tokenizer, ids, row, args.prefix_batch, prefix_logprobs=prefixes)
                        prefix_file.write(json.dumps({'task_id': row['task_id'], 'prefixes': prefixes}) + '\n')
                        result.append(from_prefixes(row, prefixes))
                return result
            calibration = score(cal)
            legacy.write_jsonl(out / 'calibration_rankings.jsonl', calibration)
            write_json(out / 'calibration.json', {key: fit_bias(calibration, key) for key in SCORE_KEYS})
            # Freeze the A-only correction before scoring any final or counterfactual target.
            for pattern in ('final_[ABCD].jsonl', 'final4_[ABCD].jsonl', 'panel_[ABCD].jsonl'):
                for path in sorted(Path(args.data).glob(pattern)):
                    rows.extend(score(read_jsonl(path)))
                    print(f'Scored {path.name}: {len(rows)} evaluation tasks', flush=True)
        (out / 'prefixes.jsonl.partial').replace(out / 'prefixes.jsonl')
        legacy.write_jsonl(out / 'rankings.jsonl', rows)
        metrics = []
        for key in SCORE_KEYS:
            computed, _ = calibrated_rows(calibration, rows, score_key=key)
            metrics.extend({**r, 'model': args.model, 'seed': args.seed, 'arm': args.arm} for r in computed)
        write_csv(out / 'task_metrics.csv', metrics)
        summary = summarize(metrics)
        write_json(out / 'hitk.json', summary)
        write_csv(out / 'summary.csv', [{k: v for k, v in r.items() if k != 'hitk'} for r in summary])
        eligible = {r['task_id'] for r in rows if all(set(x['program']) <= set(args.reliable_ops) for x in r['ranking'] if x['correct'])}
        write_json(out / 'reliable_ops_stratum.json', {'selection': 'predeclared candidate operations from historical Qwen atomic APPLY, shared across arms and families',
                   'interpretation': 'reliability must be checked against the current atomic and true-intermediate execution results',
                   'operations': args.reliable_ops, 'task_ids': sorted(eligible),
                   'summary': summarize([r for r in metrics if r['task_id'] in eligible])})
        atomic = read_jsonl(Path(args.data) / 'dev_atomic.jsonl')
        with torch.inference_mode():
            plan, plan_rows = legacy.atomic_plan_metrics(model, tokenizer, ids, atomic, args.prefix_batch)
        apply_rows = execute_programs(model, tokenizer, [{**r, 'kind': 'atomic', 'program': r['witness']} for r in atomic], min(args.prefix_batch, 8))
        intermediate = execute_programs(model, tokenizer, execution_tasks(rows, args.execution_tasks_per_family), min(args.prefix_batch, 8))
        legacy.write_jsonl(out / 'atomic_plan.jsonl', plan_rows)
        legacy.write_jsonl(out / 'execution.jsonl', [*apply_rows, *intermediate])
        cells = defaultdict(list)
        for row in [*apply_rows, *intermediate]:
            cells[row['kind'], row['program'][0] if len(row['program']) == 1 else 'multi'].append(row)
        write_json(out / 'execution_summary.json', {'atomic_plan': plan, 'cells': [
            {'kind': kind, 'operation': op, 'n': len(cell),
             'parsed': sum(r['parse_ok'] for r in cell),
             'semantic_correct': sum(r['semantic_correct'] for r in cell),
             'exact_correct': sum(r['exact_correct'] for r in cell)} for (kind, op), cell in cells.items()]})
        write_json(out / 'DONE', {'binding': binding, 'files': {
            p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('DONE', 'FAILED')}})
        (out / 'FAILED').unlink(missing_ok=True)
        print(f'Complete: {out}', flush=True)
    except Exception:
        (out / 'FAILED').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument('--prefix-batch', type=int, default=32)
    parser.add_argument('--expected-base-hash')
    parser.add_argument('--expected-adapter-hash')
    parser.add_argument('--execution-tasks-per-family', type=int, default=50)
    parser.add_argument('--reliable-ops', nargs='+', choices=core.OPS, default=['AC1', 'AX1', 'REV'])
    args = parser.parse_args()
    if args.prefix_batch < 1 or args.execution_tasks_per_family < 1:
        parser.error('Positive batch and execution panel sizes required')
    run(args)


if __name__ == '__main__':
    main()

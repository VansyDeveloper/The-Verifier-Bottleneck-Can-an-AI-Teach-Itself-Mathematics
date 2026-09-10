import argparse
import ast
import hashlib
import json
import math
import time
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import torch
import composition_core as core
import composition_eval as legacy
from .common import code_hash, environment, file_hash, read_jsonl, tree_hash, verify_data, write_json
from .modeling import PROMPT_VERSION, SCORER_ID, load


def family_summaries(metrics):
    result = {}
    for family in 'ABCD':
        rows = [row for row in metrics if row['family'] == family]
        multiple = [row for row in rows if row['correct_count'] >= 2]
        result[family] = {key: fmean(row[key] for row in rows) for key in ('hit@32', 'correct_mass', 'mrr')}
        result[family]['n_multisolution'] = len(multiple)
        for key in ('correct_conditional_entropy', 'correct_effective_count', 'max_correct_conditional_probability'):
            result[family]['multisolution_' + key] = fmean(row[key] for row in multiple) if multiple else None
    return result


def evaluate(model, tokenizer, token_ids, data, out, binding, split='dev', prefix_batch=8):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'evaluation_environment.json', environment())
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    metadata = {**binding, 'scorer_id': SCORER_ID, 'prompt_version': PROMPT_VERSION,
                'tokenizer_hash': tokenizer_hash}
    model.eval()
    metrics, accounting = [], {}
    start = time.monotonic()
    with (out / 'rankings.jsonl.partial').open('w') as raw:
        for family in 'ABCD':
            rows = read_jsonl(Path(data) / f'{split}_{family}.jsonl')
            for row in rows:
                metric, ranking = legacy.score_task(model, tokenizer, token_ids, row, prefix_batch, accounting)
                candidates = ranking['ranking']
                correct = [item for item in candidates if item['correct']]
                all_logz = legacy.logsumexp([item['score'] for item in candidates])
                correct_logz = legacy.logsumexp([item['score'] for item in correct])
                conditional = [math.exp(item['score'] - correct_logz) for item in correct]
                entropy = -sum(p * (item['score'] - correct_logz) for p, item in zip(conditional, correct))
                # JSON has no infinity; a wholly correct candidate space has no log gap.
                metric['log_gap'] = metric['log_gap'] if math.isfinite(metric['log_gap']) else None
                metric.update(metadata, family=family, degree=len(row['start']) - 1,
                              candidate_count=len(candidates), correct_count=len(correct),
                              sh1_any=any('SH1' in x['program'] for x in correct),
                              sh1_required=all('SH1' in x['program'] for x in correct),
                              program_entropy=-sum(math.exp(x['score'] - all_logz) * (x['score'] - all_logz) for x in candidates),
                              correct_conditional_entropy=entropy, correct_effective_count=math.exp(entropy),
                              max_correct_conditional_probability=max(conditional))
                metrics.append(metric)
                raw.write(json.dumps({**ranking, **metadata}, allow_nan=False) + '\n')
    (out / 'rankings.jsonl.partial').replace(out / 'rankings.jsonl')
    legacy.write_jsonl(out / 'metrics.jsonl', metrics)
    atomic_rows = read_jsonl(Path(data) / f'{split}_atomic.jsonl')
    with torch.inference_mode():
        plan, plan_raw = legacy.atomic_plan_metrics(model, tokenizer, token_ids, atomic_rows, prefix_batch)
        apply, apply_raw = legacy.atomic_apply_metrics(model, tokenizer, atomic_rows, min(prefix_batch, 8))
    per_op = defaultdict(lambda: {'n': 0, 'parsed': 0, 'semantic_correct': 0})
    for row, raw in zip(atomic_rows, apply_raw):
        parsed = None
        try:
            value = ast.literal_eval(raw['raw_completion'].strip())
            if isinstance(value, list) and len(value) == len(row['target']) and all(type(x) is int for x in value):
                parsed = value
        except (SyntaxError, ValueError, TypeError, RecursionError):
            pass
        raw['parse_ok'] = parsed is not None
        raw['semantic_correct'] = parsed is not None and [x % row['p'] for x in parsed] == row['target']
        cell = per_op[row['operation']]
        cell['n'] += 1
        cell['parsed'] += raw['parse_ok']
        cell['semantic_correct'] += raw['semantic_correct']
    for cell in per_op.values():
        cell['parse_rate'] = cell['parsed'] / cell['n']
        cell['semantic_accuracy_given_parse'] = cell['semantic_correct'] / cell['parsed'] if cell['parsed'] else None
    legacy.write_jsonl(out / 'atomic_plan.jsonl', plan_raw)
    legacy.write_jsonl(out / 'atomic_apply.jsonl', apply_raw)
    summary = {'binding': metadata, 'split': split, 'atomic_plan': plan, 'atomic_apply': apply,
               'atomic_parse': dict(per_op), 'ranking_accounting': accounting,
               'seconds': time.monotonic() - start,
               'families': family_summaries(metrics)}
    write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Exact Stage4 ranking and atomic retention')
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', choices=['dev', 'final'], default='dev')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--prefix-batch', type=int, default=8)
    args = parser.parse_args()
    if args.prefix_batch < 1:
        parser.error('--prefix-batch must be positive')
    verify_data(args.data)
    binding = {'base_hash': tree_hash(Path(args.base)),
               'adapter_hash': tree_hash(Path(args.adapter)) if args.adapter else None,
               'data_hash': file_hash(Path(args.data) / 'manifest.json'),
               'code_hash': code_hash(), 'split': args.split, 'prefix_batch': args.prefix_batch,
               'device': args.device, 'dtype': 'float32'}
    out = Path(args.out)
    if (out / 'DONE').exists():
        receipt = json.loads((out / 'DONE').read_text())
        if receipt['binding'] != binding or any(file_hash(out / name) != value for name, value in receipt['files'].items()):
            raise ValueError(f'Existing evaluation has different inputs or changed outputs: {out}')
        print(f'Already complete: {out}')
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'binding.json').exists() and json.loads((out / 'binding.json').read_text()) != binding:
        raise ValueError('Partial evaluation belongs to different inputs; use a new output directory')
    write_json(out / 'binding.json', binding)
    try:
        model, tokenizer, ids = load(args.base, adapter=args.adapter, device=args.device)
        evaluate(model, tokenizer, ids, args.data, out, binding, args.split, args.prefix_batch)
        files = {p.name: file_hash(p) for p in out.glob('*.json*') if p.name != 'DONE'}
        write_json(out / 'DONE', {'binding': binding, 'files': files})
        (out / 'FAILED').unlink(missing_ok=True)
    except Exception:
        (out / 'FAILED').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

"""Validate copied run artifacts and recompute saved results without loading weights."""

import argparse
import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

import composition_core as core
import composition_eval as legacy
from .common import file_hash, read_jsonl, tree_hash, verify_data, verify_receipt
from .modeling import PROMPT_VERSION, SCORER_ID, TARGET_TYPES


def equal(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f'{label}: expected an object')
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f'{label}: missing {key}')
            equal(actual[key], value, f'{label}/{key}')
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError(f'{label}: numeric result disagrees with raw output')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'{label}: result disagrees with raw output')


def ordered_rows(path, tasks):
    rows = read_jsonl(path)
    if [row.get('task_id') for row in rows] != [task['task_id'] for task in tasks]:
        raise ValueError(f'Task coverage/order mismatch: {path}')
    return rows


def training_budget(run, data, config):
    budget = json.loads((run / 'budget.json').read_text())
    diagnostics = budget.get('diagnostics_version')
    if diagnostics is not None:
        equal(diagnostics, 1, 'training diagnostics version')
    sources = {row['task_id']: row for name in ('train', 'atomic_train')
               for row in read_jsonl(data / f'{name}.jsonl')}
    counters = [kind + suffix for kind in ('composition', 'atomic_plan', 'atomic_apply')
                for suffix in ('_exposures', '_target_tokens')]
    totals = Counter(dict.fromkeys([*counters, *(op + '_exposures' for op in core.OPS)], 0))
    steps, widths, unique = defaultdict(Counter), defaultdict(list), set()
    component_counts, component_totals = defaultdict(Counter), Counter()
    stream = read_jsonl(run / 'training_stream.jsonl')
    for row in stream:
        task, kind, step = sources[row['task_id']], row['kind'], row['step']
        if (kind not in ('composition', 'atomic_plan', 'atomic_apply')
                or (kind == 'composition') != (task['family'] == 'TRAIN')
                or type(step) is not int or step < 1):
            raise ValueError('Invalid training stream task/kind/step')
        equal(row['set_objective'], config['method'] if config['method'] != 'ce' else None, 'training objective')
        tokens, labels = row['input_ids'], row['labels']
        if (not tokens or len(tokens) != len(labels)
                or any(type(token) is not int or token < 0 for token in tokens)
                or any(type(label) is not int or label not in (-100, token) for token, label in zip(tokens, labels))):
            raise ValueError('Invalid training tokens/labels')
        targets = sum(label != -100 for label in labels)
        if not targets:
            raise ValueError('Training exposure has no target labels')
        prompt = next(index for index, label in enumerate(labels) if label != -100)
        if any(label == -100 for label in labels[prompt:prompt + targets]):
            raise ValueError('Training target labels are not contiguous')
        if diagnostics is not None:
            types = row['target_types']
            if (len(types) != len(labels) or any(type(value) is not int or value not in range(5) for value in types)
                    or types[:prompt] != [0] * prompt or types[-1] != 3 or 3 in types[:-1]
                    or any(value == 0 for value in types[prompt:])):
                raise ValueError('Invalid training target types')
            allowed = {4} if kind == 'atomic_apply' else {1, 2}
            if not set(types[prompt:-1]) <= allowed or (config['supervision'] == 'program' and 2 in types):
                raise ValueError('Training target types disagree with supervision')
            for label, target_type in zip(labels, types):
                if label != -100:
                    component_counts[step][TARGET_TYPES[target_type - 1]] += 1
        totals.update(example_exposures=1, all_target_tokens=targets,
                      prompt_tokens=prompt, masked_target_tokens=len(tokens) - prompt - targets,
                      **{kind + '_exposures': 1, kind + '_target_tokens': targets})
        if task.get('operation'):
            totals[task['operation'] + '_exposures'] += 1
        steps[step].update(examples=1, target_tokens=targets)
        widths[step].append(len(tokens))
        if config['method'] != 'ce':
            totals['prefix_sequences'] += (5 ** task['depth'] - 1) // 4
        unique.add(row['task_id'])
    logs = read_jsonl(run / 'training_metrics.jsonl')
    if (not steps or any(type(row['step']) is not int for row in logs)
            or [row['step'] for row in logs] != list(range(1, len(steps) + 1))
            or set(steps) != set(range(1, len(steps) + 1))):
        raise ValueError('Training updates are missing or duplicated')
    for row in logs:
        equal(row, dict(steps[row['step']]), 'training update')
        for key in ('loss', 'gradient_norm', 'seconds'):
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                raise ValueError(f'Nonfinite training {key}')
        if diagnostics is not None:
            if config['method'] != 'ce':
                equal(row['ce_components'], None, 'set objective CE diagnostics')
                continue
            components = row['ce_components']
            for name in TARGET_TYPES:
                equal(components[name]['tokens'], component_counts[row['step']][name], 'CE component tokens')
                value = components[name]['loss_sum']
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid CE component loss')
                component_totals[name + '_tokens'] += components[name]['tokens']
                component_totals[name + '_loss_sum'] += value
            loss = sum(components[name]['loss_sum'] for name in TARGET_TYPES) / row['target_tokens']
            if not math.isclose(row['loss'], loss, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError('CE components disagree with optimizer loss')
    if config['method'] == 'ce':
        micro_batch = config['micro_batch']
        totals['total_forward_tokens'] = sum(max(chunk) * len(chunk) for group in widths.values()
            for offset in range(0, len(group), micro_batch) for chunk in [group[offset:offset + micro_batch]])
    totals.update(optimizer_steps=len(steps), unique_tasks=len(unique))
    equal(len(steps), config['epochs'] * math.ceil(config['num_examples'] / config['effective_batch']),
          'configured optimizer steps')
    equal({key: budget.get(key, 0) for key in totals}, dict(totals), 'training budget')
    if diagnostics is not None:
        expected = {name: {'tokens': component_totals[name + '_tokens'],
                          'loss_sum': component_totals[name + '_loss_sum']} for name in TARGET_TYPES}
        equal(budget['ce_components'], expected if config['method'] == 'ce' else None, 'CE component budget')
    return budget


def validate(run, data):
    run, data = Path(run), Path(data)
    manifest = verify_data(data)
    receipt = verify_receipt(run)
    trained = 'payload_hash' in receipt
    out = run / 'eval' if trained else run
    summary = json.loads((out / 'summary.json').read_text())
    metadata = summary['binding']
    if trained:
        config = receipt['binding']['config']
        equal(json.loads((run / 'config.resolved.json').read_text()), receipt['binding'], 'training binding')
        training = verify_receipt(run, 'TRAINED')
        training_files = ('budget.json', 'training_metrics.jsonl', 'training_stream.jsonl',
                          'reload_probe.json', 'resolved_model.json', 'environment.json')
        for name in training_files:
            if name not in training['files']:
                raise ValueError(f'Training receipt omits {name}')
        for name in (*training_files, 'config.resolved.json', 'reload_check.json'):
            if name not in receipt['files']:
                raise ValueError(f'Completion receipt omits {name}')
        payload = run / ('checkpoint' if config.get('initialize') else 'adapter')
        equal(tree_hash(payload), receipt['payload_hash'], 'saved model')
        equal(training['payload_hash'], receipt['payload_hash'], 'training payload')
        expected = {key: receipt['binding'][key] for key in ('base_hash', 'data_hash', 'code_hash')}
        expected.update(training_seed=config['seed'], method=config['method'],
                        model_hash=receipt['payload_hash'],
                        **{key: config[key] for key in ('device', 'dtype', 'prefix_batch')})
        budget = training_budget(run, data, config)
        if budget.get('diagnostics_version') == 1:
            equal(summary['split'], 'dev', 'training evaluation split')
            expected['checkpoint_step'] = budget['optimizer_steps']
            needs_midpoint = not config.get('initialize') and budget['optimizer_steps'] >= 2
            if needs_midpoint:
                midpoint = json.loads((run / 'midpoint.json').read_text())
                mid_hash = tree_hash(run / 'mid_adapter')
                equal(midpoint, {'checkpoint_step': budget['optimizer_steps'] // 2,
                                'split': 'dev', 'payload_hash': mid_hash}, 'midpoint')
                for record in (training, receipt):
                    equal(record['midpoint_payload_hash'], mid_hash, 'midpoint saved model')
                    for name in ('midpoint.json', 'mid_eval/DONE', 'mid_eval/binding.json',
                                 'mid_eval/summary.json', 'mid_eval/rankings.jsonl', 'mid_eval/metrics.jsonl',
                                 'mid_eval/atomic_plan.jsonl', 'mid_eval/atomic_apply.jsonl',
                                 'mid_eval/evaluation_environment.json'):
                        if name not in record['files']:
                            raise ValueError(f'Midpoint receipt omits {name}')
                mid_binding = json.loads((run / 'mid_eval/binding.json').read_text())
                equal(mid_binding, {**expected, 'split': 'dev', 'model_hash': mid_hash,
                                   'checkpoint_step': midpoint['checkpoint_step']}, 'midpoint provenance')
                validate(run / 'mid_eval', data)
    else:
        expected = receipt['binding']
        if 'binding.json' not in receipt['files']:
            raise ValueError('Completion receipt omits binding.json')
        equal(json.loads((out / 'binding.json').read_text()), expected, 'evaluation binding')
    equal(metadata, expected, 'evaluation provenance')
    equal(file_hash(data / 'manifest.json'), metadata['data_hash'], 'dataset manifest')
    for key in ('scorer_id', 'prompt_version', 'tokenizer_hash'):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f'Missing evaluation provenance: {key}')
    equal(metadata, {'scorer_id': SCORER_ID, 'prompt_version': PROMPT_VERSION}, 'supported evaluation protocol')
    split = summary['split']
    if split not in ('dev', 'final'):
        raise ValueError(f'Unknown evaluation split: {split}')
    if not trained:
        equal(split, expected['split'], 'evaluation split')
    required = ('summary.json', 'rankings.jsonl', 'metrics.jsonl', 'atomic_plan.jsonl',
                'atomic_apply.jsonl', 'evaluation_environment.json')
    for filename in required:
        key = f'eval/{filename}' if trained else filename
        if key not in receipt['files']:
            raise ValueError(f'Completion receipt omits {key}')
    tasks = [row for family in 'ABCD' for row in read_jsonl(data / f'{split}_{family}.jsonl')]
    if not tasks or len({row['task_id'] for row in tasks}) != len(tasks):
        raise ValueError('Empty evaluation or duplicate dataset task IDs')
    raw_rows = ordered_rows(out / 'rankings.jsonl', tasks)
    metrics = ordered_rows(out / 'metrics.jsonl', tasks)
    derived = []
    for task, raw, metric in zip(tasks, raw_rows, metrics):
        label = task['task_id']
        fingerprint = core.canonical_task_fingerprint(task['p'], task['start'], task['target'], task['depth'])
        equal(task['task_fingerprint'], fingerprint, f'{label}/fingerprint')
        record_metadata = {key: value for key, value in metadata.items() if key != 'split'}
        equal(raw, record_metadata, f'{label}/ranking provenance')
        equal(metric, record_metadata, f'{label}/metric provenance')
        # Earlier standalone evaluations stored the phase here instead of the
        # task split. Their family and exact task identity are still checked.
        if raw.get('split') not in (task['split'], metadata.get('split')):
            raise ValueError(f'{label}/ranking split disagrees with task')
        # Reuse the frozen independent enumeration/order/correctness check; its
        # historical branch/binding wrapper is not present in these new files.
        recomputed = legacy._ranking_metric_from_raw(
            {**task, 'split': raw['split']},
            {**raw, 'branch': 'iclr', 'binding': metadata}, 'iclr', metadata)
        recomputed.pop('branch')
        recomputed.pop('binding')
        recomputed['log_gap'] = recomputed['log_gap'] if math.isfinite(recomputed['log_gap']) else None
        candidates = raw['ranking']
        if any(item['score'] > 0 for item in candidates):
            raise ValueError(f'{label}: log-probability exceeds zero')
        correct = [item for item in candidates if item['correct']]
        all_logz = legacy.logsumexp([item['score'] for item in candidates])
        correct_logz = legacy.logsumexp([item['score'] for item in correct])
        probabilities = [math.exp(item['score'] - correct_logz) for item in correct]
        entropy = -sum(p * (item['score'] - correct_logz) for p, item in zip(probabilities, correct))
        recomputed.update(family=task['family'], degree=len(task['start']) - 1,
                          candidate_count=len(candidates), correct_count=len(correct),
                          sh1_any=any('SH1' in item['program'] for item in correct),
                          sh1_required=all('SH1' in item['program'] for item in correct),
                          program_entropy=-sum(math.exp(item['score'] - all_logz) *
                                               (item['score'] - all_logz) for item in candidates),
                          correct_conditional_entropy=entropy, correct_effective_count=math.exp(entropy),
                          max_correct_conditional_probability=max(probabilities))
        equal(metric, recomputed, label)
        derived.append(recomputed)
    for family in 'ABCD':
        rows = [row for row in derived if row['family'] == family]
        if not rows:
            raise ValueError(f'Missing family {family}')
        multiple = [row for row in rows if row['correct_count'] >= 2]
        values = {key: fmean(row[key] for row in rows) for key in ('hit@32', 'correct_mass', 'mrr')}
        values['n_multisolution'] = len(multiple)
        for key in ('correct_conditional_entropy', 'correct_effective_count', 'max_correct_conditional_probability'):
            values['multisolution_' + key] = fmean(row[key] for row in multiple) if multiple else None
        equal(summary['families'][family], values, f'family {family}')
    atomic = read_jsonl(data / f'{split}_atomic.jsonl')
    plan = ordered_rows(out / 'atomic_plan.jsonl', atomic)
    apply = ordered_rows(out / 'atomic_apply.jsonl', atomic)
    counts = defaultdict(lambda: dict(n=0, plan=0, apply=0, parsed=0, semantic_correct=0))
    for task, prediction, completion in zip(atomic, plan, apply):
        op = task['operation']
        target = list(core.trajectory(task['start'], [op], task['p'])[-1])
        equal(task['target'], target, f"{task['task_id']}/atomic target")
        if prediction['plan_prediction'] not in core.OPS:
            raise ValueError('Invalid atomic PLAN prediction')
        plan_ok = prediction['plan_prediction'] == op
        equal(prediction, {'operation': op, 'plan_correct': plan_ok}, 'atomic PLAN')
        text = completion['raw_completion']
        if not isinstance(text, str):
            raise ValueError('Invalid atomic APPLY completion')
        apply_ok = text.strip() == core.format_state(target)
        parsed = None
        try:
            value = ast.literal_eval(text.strip())
            if isinstance(value, list) and len(value) == len(target) and all(type(x) is int for x in value):
                parsed = value
        except (SyntaxError, ValueError, TypeError, RecursionError):
            pass
        semantic_ok = parsed is not None and [value % task['p'] for value in parsed] == target
        equal(completion, {'operation': op, 'target': core.format_state(target), 'correct': apply_ok,
                           'match_mode': 'stripped_exact_equality', 'parse_ok': parsed is not None,
                           'semantic_correct': semantic_ok}, 'atomic APPLY')
        cell = counts[op]
        for key, value in dict(n=1, plan=plan_ok, apply=apply_ok, parsed=parsed is not None,
                               semantic_correct=semantic_ok).items():
            cell[key] += value
    if set(counts) != set(core.OPS):
        raise ValueError('Atomic evaluation omits an operation')
    for kind in ('plan', 'apply'):
        equal(summary['atomic_' + kind], {
            'overall': sum(cell[kind] for cell in counts.values()) / len(atomic),
            'by_operation': {op: cell[kind] / cell['n'] for op, cell in counts.items()}}, f'atomic {kind} summary')
    for op, cell in counts.items():
        expected = {key: cell[key] for key in ('n', 'parsed', 'semantic_correct')}
        expected.update(parse_rate=cell['parsed'] / cell['n'],
                        semantic_accuracy_given_parse=cell['semantic_correct'] / cell['parsed'] if cell['parsed'] else None)
        equal(summary['atomic_parse'][op], expected, f'atomic {op} parse summary')
    return {'status': 'PASS', 'run': str(run.resolve()), 'data_status': manifest['status'],
            'split': split, 'composition_tasks': len(tasks), 'atomic_tasks': len(atomic),
            'raw_results_recomputed': True, 'payload_hash_checked': trained,
            'training_budget_checked': trained, 'source_code_hash': metadata['code_hash'],
            'midpoint_checked': trained and budget.get('diagnostics_version') == 1 and needs_midpoint,
            'input_receipt_sha256': file_hash(run / 'DONE'), 'data_hash': metadata['data_hash'],
            'model_inference_repeated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='completed training run or standalone evaluation')
    parser.add_argument('--data', type=Path, required=True, help='dataset directory, including after copying outputs')
    args = parser.parse_args()
    print(json.dumps(validate(args.run, args.data), indent=2))


if __name__ == '__main__':
    main()

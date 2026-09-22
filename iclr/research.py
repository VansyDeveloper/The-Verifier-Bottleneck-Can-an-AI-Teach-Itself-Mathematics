"""V4 Q1-Q4 queues; explicit dev gates, small confirmation, immutable final lock."""

import argparse
import json
from pathlib import Path
import shutil

from .common import ROOT, code_hash, file_hash, tree_hash, verify_data, verify_receipt, write_json
from .run import write_config
from .upgrade import run_queue, send_results, verify_bundle, verify_job
from .research_io import exposure_ledger

SCOPES = {
    'Q1': 'Existing checkpoints: crossed and ordinary conditional evaluation',
    'Q2': 'Exact/MC gradient diagnosis, then gated reward x entropy pilot',
    'Q3': 'Matched CE/counterfactual x entropy pilot',
    'Q4': 'State-access diagnosis, then gated state/SIGReg pilot',
    'confirm': 'Two selected conditions x two masks x three paired continuation seeds',
    'external': 'Selected pair on a small nonlinear DSL; not a published benchmark',
    'budget': 'Selected supervised pair at one prespecified dev budget, same initialization',
}
ARMS = {'Q2': ['sampled', 'sampled_entropy', 'exact', 'exact_entropy'],
        'Q3': ['ce', 'ce_entropy', 'ce_cf', 'ce_cf_entropy'],
        'Q4': ['state_ce', 'state', 'sigreg', 'state_sigreg']}


def verify_inputs(path):
    path = Path(path)
    protocol = json.loads((path / 'protocol.json').read_text())
    if (protocol['schema'] != 'iclr.research.protocol.v4'
            or protocol['analysis_plan_sha256'] != file_hash(ROOT / 'plans/research_v4.json')
            or protocol['audit_sha256'] != file_hash(path / 'audit.json')):
        raise ValueError('V4 protocol/plan/audit binding failed')
    if json.loads((path / 'audit.json').read_text())['status'] != 'PASS':
        raise ValueError('CPU dataset audit did not pass')
    for name, entry in protocol['datasets'].items():
        if entry['manifest_sha256'] != file_hash(path / name / 'manifest.json'):
            raise ValueError('V4 dataset manifest changed')
        verify_data(path / name)
    return protocol


def import_inputs(args):
    source, out = Path(args.inputs), Path(args.out)
    verify_inputs(source)
    if file_hash(source / 'protocol.json') != args.expected_hash:
        raise ValueError('Shared inputs changed after planning')
    if out.exists():
        verify_inputs(out)
        if file_hash(out / 'protocol.json') != args.expected_hash:
            raise ValueError('Queue data differ from frozen inputs')
        return
    temp = out.with_name(out.name + '.copying')
    shutil.copytree(source, temp)
    verify_inputs(temp)
    temp.rename(out)


def plan(args):
    protocol = verify_inputs(args.inputs)
    if protocol['smoke'] != args.smoke:
        raise ValueError('Use matching --smoke and input status')
    if args.phase == 'final' and (not args.analysis_lock or args.queue_name != 'Q1' and not args.trained_from):
        raise ValueError('Final requires --analysis-lock and --trained-from; no training is run again')
    if args.queue_name in ('Q1', 'Q3') and args.stage != 'screen':
        raise ValueError('Q1/Q3 have only the screen stage; use confirm after explicit dev selection')
    if args.queue_name in ('Q2', 'Q4') and args.stage == 'pilot' and not args.gate:
        raise ValueError('Pilot requires --gate with the completed matching diagnostic')
    if args.queue_name == 'external' and args.atomic_epochs < 1:
        raise ValueError('Atomic warm-up needs at least one epoch')
    profiles = json.loads(Path(args.models).read_text())
    models = [m.copy() for m in profiles if m['name'] in args.model_names]
    if {m['name'] for m in models} != set(args.model_names) or not models:
        raise ValueError('Selected model profile is missing')
    if args.queue_name != 'Q1' and len(models) != 1:
        raise ValueError('Screen one model/initialization; do not multiply the pilot grid')
    tuning = {key: getattr(args, key) for key in ('cf_weight', 'tau', 'entropy_weight',
              'state_weight', 'sigreg_weight', 'learning_rate', 'epochs') if getattr(args, key, None) is not None}
    if tuning and (args.phase == 'final' or args.queue_name in ('confirm', 'external')):
        raise ValueError('Tune only on dev pilots; confirmation/final inherit the locked dev choice')
    selection = None
    if args.queue_name in ('confirm', 'external', 'budget'):
        if not args.selection:
            raise ValueError('Confirmation/external replication needs an explicit dev --selection')
        selection = json.loads(Path(args.selection).read_text())
        if selection['code_hash'] != code_hash() or selection['plan_hash'] != file_hash(ROOT / 'plans/research_v4.json'):
            raise ValueError('Dev selection belongs to different code/plan')
        if args.phase == 'final':
            lock = json.loads(Path(args.analysis_lock).read_text())
            if file_hash(args.selection) not in lock['selection_hashes']:
                raise ValueError('Final selection differs from the frozen dev choice')
    family = selection['family'] if selection else args.queue_name
    if args.queue_name == 'budget' and (args.phase != 'dev' or family not in ('Q3', 'Q4')
            or set(tuning) != {'epochs'} or args.epochs not in (1, 2, 4)):
        raise ValueError('Budget curve: selected supervised pair, dev only, --epochs 1, 2 or 4; no other tuning')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs, data = [], out / 'data'
    def add(name, module, config, kind='receipt', depends=('inputs',), extra=()):
        path = out / 'configs' / (name + '.json')
        write_config(path, config)
        jobs.append({'id': name, 'argv': ['-m', module, '--config', str(path), *extra],
                     'config_path': str(path), 'config_sha256': file_hash(path),
                     'result': config['output'], 'kind': kind, 'depends_on': list(depends),
                     'resource': 'gpu', 'seed': config.get('seed'), 'experiment': args.queue_name})
    jobs.append({'id': 'inputs', 'argv': ['-m', 'iclr.research', 'import-inputs', '--inputs', str(args.inputs),
        '--out', str(data), '--expected-hash', file_hash(Path(args.inputs) / 'protocol.json')],
        'result': str(data), 'kind': 'data', 'depends_on': [], 'resource': 'cpu', 'experiment': args.queue_name})
    for model in models:
        history = json.loads(Path(model['receipt']).read_text()) if model.get('receipt') else {'runs': []}
        if history['runs'] and args.queue_name != 'external' and {r['data_hash'] for r in history['runs']} != {protocol['reference_manifest_sha256']}:
            raise ValueError('Model history and inherited data differ')
        base_hash = history['runs'][0]['base_hash'] if history['runs'] else model.get('base_hash')
        if selection and (selection['initialization']['model'] != model['model']
                or selection['initialization']['base_hash'] != (base_hash or tree_hash(model['base']))
                or selection['initialization']['dtype'] != args.dtype):
            raise ValueError('Dev-selected method must retain its model, initialization, and precision')
        model['required_adapters'] = []
        common = dict(base=model['base'], model=model['model'], device=args.device, dtype=args.dtype,
                      prefix_batch=args.prefix_batch, expected_base_hash=base_hash, seed=0)
        if args.queue_name == 'Q1':
            runs = [('baseline', None, None, -1)] + [(r['arm'], str(Path(model['previous']) / r['run'] / 'adapter'),
                       r['adapter_hash'], r['seed']) for r in history['runs']]
            for arm, adapter, digest, seed in runs:
                name = f'{model["name"]}_{arm}_seed{seed}'
                if adapter:
                    model['required_adapters'].append(adapter)
                config = {**common, 'data': str(data / 'original'), 'adapter': adapter,
                    'expected_adapter_hash': digest, 'arm': arm, 'seed': seed, 'phase': args.phase,
                    'output': str(out / 'evaluations' / name), 'analysis_lock': args.analysis_lock}
                add(name, 'iclr.research_evaluate', config)
            continue
        reward = args.queue_name == 'Q2' or selection and selection['family'] == 'Q2'
        if reward:
            previous = next((r for r in history['runs'] if r['arm'] == 'composition' and r['seed'] == 0), None)
            if previous is None and not args.smoke:
                raise ValueError('Q2 requires the frozen historical composition seed0 checkpoint')
            if previous:
                adapter = str(Path(model['previous']) / previous['run'] / 'adapter')
                common.update(adapter=adapter, expected_adapter_hash=previous['adapter_hash'])
        if selection and not reward and selection['initialization']['adapter_hash'] != common.get('expected_adapter_hash'):
            raise ValueError('Selected method must retain its initial continuation adapter')
        default_mask = selection['dataset'] if args.queue_name == 'budget' else 'original' if reward else 'mask1'
        masks = ['mask1', 'mask2'] if args.queue_name == 'confirm' else ['listdsl'] if args.queue_name == 'external' else [args.mask or default_mask]
        seeds = [0, 1, 2] if args.queue_name == 'confirm' else [0]
        if args.queue_name == 'external' and protocol.get('domain') != 'nonlinear_list_v1':
            raise ValueError('External queue requires the frozen nonlinear list DSL inputs')
        for mask in masks:
            if mask not in protocol['datasets']:
                raise ValueError(f'Missing dataset {mask}')
            source = {**common, 'data': str(data / mask)}
            gate, dependencies = args.gate, ['inputs']
            if reward and mask not in ('original', 'listdsl'):
                receipt_path = model.get('reward_initializations', {}).get(mask)
                if not receipt_path and getattr(args, 'prepare_matching_sft', False):
                    origin = Path(args.trained_from) if args.phase == 'final' else out
                    receipt_path = str(origin / 'training' / f'{model["name"]}_{mask}_sft_init')
                    if args.phase == 'dev':
                        name = f'{model["name"]}_{mask}_sft_init'
                        add(name, 'iclr.research_train', {**source, 'adapter': None, 'expected_adapter_hash': None,
                            'objective': 'ce', 'epochs': 1 if args.smoke else 2, 'output': receipt_path}, kind='train')
                        dependencies = [name]
                        source.update(adapter=str(Path(receipt_path) / 'adapter'), expected_adapter_hash=None,
                                      initial_training_receipt=receipt_path)
                elif not receipt_path:
                    raise ValueError(f'Q2 {mask} needs matching-mask SFT: use --prepare-matching-sft or reward_initializations[{mask}] with its completed training receipt')
                if not (getattr(args, 'prepare_matching_sft', False) and args.phase == 'dev' and dependencies != ['inputs']):
                    prior = verify_receipt(receipt_path)
                    prior_data = Path(prior['binding']['config']['data'])
                    prior_manifest = verify_data(prior_data)
                    stages = json.loads((Path(receipt_path) / 'exposure.json').read_text())['stages']
                    target_manifest = json.loads((Path(args.inputs) / mask / 'manifest.json').read_text())
                    if (file_hash(prior_data / 'manifest.json') != prior['binding']['data_hash']
                        or prior['binding']['config'].get('objective') != 'ce'
                        or prior_manifest['constraints'] != target_manifest['constraints']
                        or prior['binding']['base_hash'] != (base_hash or tree_hash(model['base']))
                        or not exposure_ledger(target_manifest, stages)['recorded_history_excludes_mask']):
                        raise ValueError('Reward initialization history does not exclude this mask')
                    verify_job({'kind': 'train', 'result': receipt_path})
                    adapter = str(Path(receipt_path) / 'adapter')
                    source.update(adapter=adapter, expected_adapter_hash=prior['payload_hash'], initial_training_receipt=str(receipt_path))
            if source.get('adapter') and dependencies == ['inputs']:
                model['required_adapters'].append(source['adapter'])
            if args.queue_name == 'external':
                origin = Path(args.trained_from) if args.phase == 'final' else out
                warmup = origin / 'training' / f'{model["name"]}_listdsl_atomic'
                atomic_gate = origin / 'diagnostics' / f'{model["name"]}_listdsl_atomic'
                if args.phase == 'dev':
                    add('listdsl_atomic_train', 'iclr.research_train', {**source, 'objective': 'atomic',
                        'epochs': 1 if args.smoke else args.atomic_epochs, 'output': str(warmup)}, kind='train')
                source.update(adapter=str(warmup / 'adapter'), expected_adapter_hash=None,
                              initial_training_receipt=str(warmup))
                if args.phase == 'dev':
                    add('listdsl_atomic_check', 'iclr.research_evaluate', {**source, 'output': str(atomic_gate)},
                        depends=['listdsl_atomic_train'], extra=('--atomic-check',))
                    dependencies = ['listdsl_atomic_check']
                source['atomic_gate'] = str(atomic_gate)
            if family in ('Q2', 'Q4') and (args.stage == 'screen' or args.queue_name in ('confirm', 'external', 'budget')) and args.phase == 'dev':
                name = f'{model["name"]}_{mask}_diagnostic'
                gate = str(out / 'diagnostics' / name)
                config = {**source, 'output': gate}
                if args.smoke:
                    config.update(**({'train_tasks': 1, 'dev_panels_per_family': 1, 'mc_groups': 16, 'mc_max_groups': 16, 'mc_blocks': 4,
                                      'learning_rates': [1e-5, 1e-6]} if family == 'Q2' else {'panels_per_family': 1}))
                add(name, 'iclr.research_gradient' if family == 'Q2' else 'iclr.research_evaluate', config,
                    depends=dependencies,
                    extra=('--state-access',) if family == 'Q4' else ())
                dependencies = [name]
                if args.queue_name not in ('confirm', 'external', 'budget'):
                    continue
            arms = selection['arms'] if selection else ARMS[family]
            baseline_name = f'{model["name"]}_{mask}_initial'
            baseline_config = {**source, 'arm': 'initial', 'seed': -1, 'phase': args.phase,
                'analysis_lock': args.analysis_lock, 'output': str(out / 'evaluations' / baseline_name)}
            if args.smoke:
                baseline_config['solve_per_family'] = 1
            add(baseline_name, 'iclr.research_evaluate', baseline_config, depends=dependencies)
            dependencies = [baseline_name]
            for seed in seeds:
                for arm in arms:
                    name = f'{model["name"]}_{mask}_{arm}_seed{seed}'
                    directory = out / 'training' / name
                    settings = selection['settings'][arm] if selection else {}
                    training = {**settings, **source, **tuning, 'output': str(directory), 'objective': arm, 'seed': seed,
                        'epochs': 1 if args.smoke else tuning.get('epochs', settings.get('epochs', 2)), 'gate': gate}
                    if args.smoke:
                        training['reward_steps'] = 1
                    if args.phase == 'dev':
                        add(name + '_train', 'iclr.research_train', training, kind='train', depends=dependencies)
                        eval_dependencies = [name + '_train']
                        adapter_hash = None
                    else:
                        directory = Path(args.trained_from) / 'training' / name
                        receipt = verify_receipt(directory)
                        bound = receipt['binding']
                        if (bound['code_hash'] != code_hash() or bound['data_hash'] != protocol['datasets'][mask]['manifest_sha256']
                                or any(bound['config'][k] != training[k] for k in ('objective', 'seed', 'epochs'))
                                or base_hash and bound['base_hash'] != base_hash
                                or tree_hash(directory / 'adapter') != receipt['payload_hash']):
                            raise ValueError('Final checkpoint differs from the selected training cell')
                        adapter_hash, eval_dependencies = receipt['payload_hash'], ['inputs']
                    evaluate = {**source, 'seed': seed, 'arm': arm, 'adapter': str(directory / 'adapter'),
                        'training_receipt': str(directory),
                        'expected_adapter_hash': adapter_hash, 'phase': args.phase, 'analysis_lock': args.analysis_lock,
                        'output': str(out / 'evaluations' / name)}
                    if args.smoke:
                        evaluate['solve_per_family'] = 1
                    add(name + '_eval', 'iclr.research_evaluate', evaluate, depends=eval_dependencies)
    queue = {'schema': 'iclr.research.queue.v4', 'source_code_hash': code_hash(),
        'research_plan_path': 'plans/research_v4.json', 'family': family,
        'research_plan_sha256': file_hash(ROOT / 'plans/research_v4.json'), 'models': models, 'jobs': jobs,
        'phase': args.phase, 'stage': args.stage, 'smoke': args.smoke, 'experiments': [args.queue_name],
        'scope': {args.queue_name: SCOPES[args.queue_name]}, 'protocol_hash': file_hash(Path(args.inputs) / 'protocol.json')}
    queue['selection_hash'] = file_hash(args.selection) if args.selection else None
    queue['selection_path'] = str(Path(args.selection).resolve()) if args.selection else None
    queue['analysis_lock_path'] = str(Path(args.analysis_lock).resolve()) if args.analysis_lock else None
    queue['analysis_lock_sha256'] = file_hash(args.analysis_lock) if args.analysis_lock else None
    write_config(out / 'queue.json', queue)
    if not (out / 'status.json').exists():
        write_json(out / 'status.json', {'queue_sha256': file_hash(out / 'queue.json'),
            'jobs': {job['id']: {'status': 'planned'} for job in jobs}})
    print(json.dumps({'queue': str(out / 'queue.json'), 'jobs': len(jobs),
                      'training_jobs': sum(j['kind'] == 'train' for j in jobs), 'phase': args.phase}))


def select(args):
    source = Path(args.queue)
    queue = json.loads(source.read_text())
    if queue['source_code_hash'] != code_hash() or queue['research_plan_sha256'] != file_hash(ROOT / 'plans/research_v4.json'):
        raise ValueError('Select from dev results produced by this exact code/plan')
    family = queue.get('family', queue['experiments'][0])
    if (queue['experiments'][0] not in (*ARMS, 'budget') or family not in ARMS
            or args.method not in ARMS[family] or args.method == ARMS[family][0] or queue['phase'] != 'dev'):
        raise ValueError('Select one non-baseline method from a completed dev pilot')
    arms = [ARMS[family][0], args.method]
    settings, hashes, initializations = {}, {}, set()
    for job in queue['jobs']:
        if job['kind'] == 'train':
            receipt = verify_receipt(job['result'])
            cfg = receipt['binding']['config']
            if cfg['objective'] in arms:
                verify_job(job)
                initializations.add((cfg['model'], receipt['binding']['base_hash'],
                                     receipt['binding']['adapter_hash'], cfg['dtype']))
                settings[cfg['objective']] = {k: v for k, v in cfg.items() if k not in
                    ('base', 'adapter', 'model', 'data', 'output', 'expected_base_hash', 'expected_adapter_hash',
                     'gate', 'atomic_gate', 'initial_training_receipt')}
                dataset = Path(cfg['data']).name
                hashes[job['id']] = file_hash(Path(job['result']) / 'DONE')
        elif job['kind'] == 'receipt':
            hashes[job['id']] = verify_job(job)
    if set(settings) != set(arms) or len(initializations) != 1:
        raise ValueError('Missing completed paired training cells')
    model, base_hash, adapter_hash, dtype = initializations.pop()
    write_config(Path(args.out), {'schema': 'iclr.research.selection.v4', 'family': family, 'arms': arms,
        'settings': settings, 'dataset': dataset, 'source_queue': str(source.resolve()),
        'source_queue_sha256': file_hash(source), 'source_receipts': hashes,
        'initialization': {'model': model, 'base_hash': base_hash, 'adapter_hash': adapter_hash, 'dtype': dtype},
        'code_hash': code_hash(), 'plan_hash': file_hash(ROOT / 'plans/research_v4.json'),
        'reason': args.reason, 'scope': 'dev choice; final outcomes were not opened'})


def freeze(args):
    queues = [json.loads(Path(p).read_text()) for p in args.queues]
    checkpoints, datasets = set(), set()
    for queue in queues:
        if queue['phase'] != 'dev' or queue['source_code_hash'] != code_hash():
            raise ValueError('Freeze completed dev queues from this code version')
        for job in queue['jobs']:
            verify_job(job)
            if job['kind'] == 'receipt':
                binding = verify_receipt(job['result'])['binding']
                if binding['kind'] == 'evaluate':
                    checkpoints.add((binding['base_hash'], binding['adapter_hash']))
                    datasets.add(binding['data_hash'])
    if not checkpoints:
        raise ValueError('No completed dev evaluations to freeze')
    write_config(Path(args.out), {'schema': 'iclr.research.lock.v4', 'code_hash': code_hash(),
        'plan_hash': file_hash(ROOT / 'plans/research_v4.json'), 'dataset_hashes': sorted(datasets),
        'checkpoints': sorted(checkpoints, key=str), 'queues': {p: file_hash(p) for p in args.queues},
        'selection_hashes': sorted({q['selection_hash'] for q in queues if q.get('selection_hash')}),
        'checkpoint_rule': 'fixed final saved adapter; no retraining and no final selection'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for action in ('plan', 'start'):
        p = sub.add_parser(action)
        p.add_argument('--queue-name', choices=SCOPES, required=True)
        p.add_argument('--stage', choices=['screen', 'pilot'], default='screen')
        p.add_argument('--inputs', required=True)
        p.add_argument('--models', default='plans/upgrade_models.json')
        p.add_argument('--model-names', nargs='+', default=['q06'])
        p.add_argument('--mask', choices=['original', 'mask1', 'mask2'])
        p.add_argument('--epochs', type=int)
        p.add_argument('--atomic-epochs', type=int, default=2)
        p.add_argument('--prepare-matching-sft', action='store_true', help='Q2: prepare one shared CE initialization per new mask before reward diagnostics')
        p.add_argument('--phase', choices=['dev', 'final'], default='dev')
        p.add_argument('--analysis-lock')
        p.add_argument('--trained-from')
        p.add_argument('--gate')
        p.add_argument('--selection')
        p.add_argument('--out', required=True)
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
        p.add_argument('--prefix-batch', type=int, default=16)
        for name in ('cf-weight', 'tau', 'entropy-weight', 'state-weight', 'sigreg-weight', 'learning-rate'):
            p.add_argument('--' + name, type=float)
        p.add_argument('--smoke', action='store_true')
        if action == 'start':
            p.add_argument('--gpus', nargs='+', required=True)
            p.add_argument('--cpu-threads', type=int, default=4)
            p.add_argument('--retry-failed', action='store_true')
    p = sub.add_parser('import-inputs')
    p.add_argument('--inputs', required=True); p.add_argument('--out', required=True); p.add_argument('--expected-hash', required=True)
    p = sub.add_parser('run')
    p.add_argument('--queue', required=True); p.add_argument('--gpus', nargs='+', required=True)
    p.add_argument('--cpu-threads', type=int, default=4); p.add_argument('--retry-failed', action='store_true')
    p = sub.add_parser('select')
    p.add_argument('--queue', required=True); p.add_argument('--method', required=True)
    p.add_argument('--reason', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--queues', nargs='+', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('send')
    p.add_argument('--out', required=True); p.add_argument('--destination'); p.add_argument('--allow-incomplete', action='store_true')
    p = sub.add_parser('verify-bundle'); p.add_argument('--archive', required=True)
    args = parser.parse_args()
    if args.command in ('plan', 'start'):
        plan(args)
        if args.command == 'plan':
            return
        args.queue = str(Path(args.out) / 'queue.json')
    if args.command in ('start', 'run'):
        args.send_on_complete = True
        run_queue(args)
    else:
        {'import-inputs': import_inputs, 'select': select, 'freeze': freeze,
         'send': send_results, 'verify-bundle': verify_bundle}[args.command](args)


if __name__ == '__main__':
    main()

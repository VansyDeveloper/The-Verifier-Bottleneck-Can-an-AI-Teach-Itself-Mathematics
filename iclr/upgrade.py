"""Additive ICLR queue: one process per GPU, resumable receipts, portable export."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile

from .common import ROOT, code_hash, file_hash, tree_hash, verify_data, verify_receipt, write_json
from .run import write_config

EXPERIMENTS = {
    '01_recheck_existing_models': 'Новая проверка уже обученных Qwen0.6 и Qwen8B',
    '02_new_pair_masks': 'Четыре новые маски исключённых пар',
    '03_new_triples_and_positions': 'Новая тройка и знакомая пара на новой позиции',
    '04_correct_program_choice': 'Выбор правильного свидетеля и MML',
    '05_reward_gradient': 'Сэмплированный и точный reward-градиент',
    '06_other_model_family': 'Другая модельная семья с парным контролем Qwen',
}


def experiment_of(name):
    number = (5 if 'reward' in name else 6 if name.startswith('smol17_') or '_original_' in name
              else 2 if '_mask' in name else 3 if '_triple_' in name or '_position_' in name
              else 4 if '_witness_' in name else 1)
    return list(EXPERIMENTS)[number - 1]


def plan(args):
    if getattr(args, 'inputs', None):
        if getattr(args, 'prepared_data', None) or args.reference != 'outputs/data':
            raise ValueError('Use --inputs or separate --reference/--prepared-data paths, not both')
        args.reference = str(Path(args.inputs) / 'reference_data')
        args.prepared_data = str(Path(args.inputs) / 'new_tasks')
    experiments = list(getattr(args, 'experiments', None) or list(EXPERIMENTS)[:4])
    experiments += ([list(EXPERIMENTS)[4]] if args.include_grpo else []) + ([list(EXPERIMENTS)[5]] if args.include_external else [])
    experiments = list(dict.fromkeys(experiments))
    include_grpo = list(EXPERIMENTS)[4] in experiments
    include_external = list(EXPERIMENTS)[5] in experiments
    seeds = getattr(args, 'seeds', [0, 1, 2])
    if ((include_grpo and not {0, 1} <= set(seeds)) or
            (include_external and set(seeds) != {0, 1, 2})):
        raise ValueError('Keep experiments 05 and 06 whole on one computer; they share pilot gates or a fresh atomic initialization')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    models = json.loads(Path(args.models).read_text())
    if any(e != list(EXPERIMENTS)[0] for e in experiments) and not any(m['name'] == 'q06' for m in models):
        raise ValueError('The selected new training blocks require the q06 inherited model profile')
    if include_external:
        external = json.loads((ROOT / 'evidence/upgrade/external_model.json').read_text())
        models.append({'name': 'smol17', 'model': external['model'], 'revision': external['revision'],
                       'base': str(output / 'experiments/06_other_model_family/atomic_initialization/checkpoint')})
    jobs, configs = [], {}
    def add(name, module, arguments, result, kind='receipt', dependencies=(), resource='gpu'):
        seed = re.search(r'_seed(\d+)(?:_closed)?$', name)
        jobs.append({'id': name, 'argv': ['-m', module, *map(str, arguments)],
                     'result': str(result), 'kind': kind, 'depends_on': list(dependencies), 'resource': resource,
                     'experiment': experiment_of(name), 'seed': int(seed[1]) if seed else None,
                     'priority': 0 if resource == 'cpu' else 6 if name.startswith('smol17_') or '_original_' in name
                     else 1 if '_previous_' in name or name.endswith('_baseline')
                     else 5 if '_reward_' in name else 4 if '_witness_' in name
                     else 3 if '_triple_' in name or '_position_' in name else 2})
    data = output / 'data'
    if getattr(args, 'prepared_data', None):
        protocol = Path(args.prepared_data) / 'protocol.json'
        if json.loads(protocol.read_text())['smoke'] != args.smoke:
            raise ValueError('Prepared data smoke/full status differs from the requested queue')
        add('prepare_data', 'iclr.upgrade', ['import-data', '--source', args.prepared_data,
            '--reference', args.reference, '--out', data, '--expected-protocol-hash', file_hash(protocol)],
            data, 'data', resource='cpu')
    else:
        add('prepare_data', 'iclr.upgrade_data', ['--reference', args.reference, '--out', data,
            *(['--smoke'] if args.smoke else [])], data, 'data', resource='cpu')
    add('audit_data', 'iclr.upgrade_audit', ['--reference', args.reference, '--data', data,
        '--out', output / 'data_audit'], output / 'data_audit', dependencies=['prepare_data'], resource='cpu')
    if include_grpo:
        add('reward_gradient_check', 'iclr.reward_control', ['--check', '--out', output / 'reward_gradient_check'],
            output / 'reward_gradient_check', resource='cpu')
    for model in models:
        history = json.loads(Path(model['receipt']).read_text()) if model.get('receipt') else {'runs': []}
        base_hash = history['runs'][0]['base_hash'] if history['runs'] else None
        base_dependencies = ['audit_data']
        if not history['runs']:
            name = model['name'] + '_atomic_init'
            directory = Path(model['base']).parent
            config = {'initialize': True, 'model': model['model'], 'revision': model['revision'],
                'data': str(data / 'original'), 'output': str(directory), 'seed': 0,
                'replay_fraction': 1., 'epochs': 1 if args.smoke else 2,
                'num_examples': 4 if args.smoke else 10000,
                'effective_batch': 4 if args.smoke else 64, 'micro_batch': 1 if args.smoke else 16,
                'prefix_batch': args.prefix_batch, 'device': args.device, 'dtype': args.dtype}
            path = output / 'configs' / (name + '.json')
            configs[path] = config
            add(name, 'iclr.train', ['--config', path], directory, 'init', base_dependencies)
            base_dependencies = [name]
        def evaluate(name, dataset, arm, seed, adapter=None, adapter_hash=None, dependencies=None):
            directory = output / 'experiments' / experiment_of(name) / 'evaluations' / name
            arguments = ['--model', model['model'], '--base', model['base'], '--data', data / dataset,
                '--out', directory, '--seed', seed, '--arm', arm,
                '--device', args.device, '--dtype', args.dtype, '--prefix-batch', args.prefix_batch]
            if base_hash:
                arguments += ['--expected-base-hash', base_hash]
            if adapter:
                arguments += ['--adapter', adapter]
            if adapter_hash:
                arguments += ['--expected-adapter-hash', adapter_hash]
            if args.smoke:
                arguments += ['--execution-tasks-per-family', 1]
            add(name, 'iclr.upgrade_evaluate', arguments, directory,
                dependencies=base_dependencies if dependencies is None else dependencies)
        evaluate(model['name'] + '_baseline', 'original', 'baseline', -1)
        for previous in history['runs']:
            name = f"{model['name']}_previous_{previous['arm']}_seed{previous['seed']}"
            adapter = Path(model['previous']) / previous['run'] / 'adapter'
            evaluate(name, 'original', previous['arm'], previous['seed'], adapter, previous['adapter_hash'])
        # The source plan prioritizes the structural experiment on one size:
        # four masks x two seeds x two arms, before another model-size sweep.
        if model['name'] not in ('q06', 'smol17'):
            continue
        cells = []
        for dataset in (('mask1', 'mask2', 'mask3', 'mask4', 'triple', 'position') if model['name'] == 'q06' else ()):
            for seed in (0, 1):
                for arm, replay, budget in (('atomic_control', 1., 'target_tokens'), ('composition', .2, 'examples')):
                    cells.append((dataset, arm, seed, {'replay_fraction': replay, 'budget_mode': budget, 'supervision': 'trace'}))
        if include_external:
            for seed in (0, 1, 2):
                for arm, replay, budget in (('atomic_control', 1., 'target_tokens'), ('composition', .2, 'examples')):
                    cells.append(('original', arm, seed, {'replay_fraction': replay, 'budget_mode': budget, 'supervision': 'trace'}))
        for seed in ((0, 1, 2) if model['name'] == 'q06' else ()):
            for arm, method, witness in (('fixed', 'ce', 'fixed'), ('uniform', 'ce', 'uniform'),
                                         ('balanced', 'ce', 'balanced'), ('single_norm', 'single_norm', 'fixed'),
                                         ('mml', 'set_mass', 'fixed')):
                cells.append(('witness', arm, seed, {'method': method, 'witness_policy': witness,
                    'supervision': 'program', 'depth3_only': True, 'replay_fraction': 0.,
                    'lora_dropout': 0., 'budget_mode': 'examples'}))
        for dataset, arm, seed, cell in cells:
            name = f"{model['name']}_{dataset}_{arm}_seed{seed}"
            directory = output / 'experiments' / experiment_of(name) / 'training' / name
            config = {'model': model['model'], 'base': model['base'], 'data': str(data / dataset),
                'output': str(directory), 'seed': seed, 'epochs': 1 if args.smoke else 2,
                # Cover every matched composition task once, plus 20% atomic replay.
                # A 5000-exposure mixture would discard 1000 tasks and their matched marginals.
                'num_examples': 5000 if dataset == 'witness' else 6250,
                'effective_batch': 4 if args.smoke else 64, 'micro_batch': 1 if args.smoke else 16,
                'prefix_batch': args.prefix_batch, 'device': args.device, 'dtype': args.dtype,
                'learning_rate': 1e-4, **cell}
            if args.smoke:
                config['num_examples'] = 4
            path = output / 'configs' / (name + '.json')
            configs[path] = config
            add(name, 'iclr.train', ['--config', path], directory, 'train', base_dependencies)
            evaluate(name + '_closed', dataset, arm, seed, directory / 'adapter', dependencies=(name,))
        if include_grpo and model['name'] == 'q06':
            initial = next(r for r in history['runs'] if r['arm'] == 'composition' and r['seed'] == 0)
            initial_adapter = Path(model['previous']) / initial['run'] / 'adapter'
            pilots = [f"{model['name']}_reward_pilot_{method}_seed{seed}"
                      for method in ('sampled', 'exact') for seed in (0, 1)]
            for phase, optimizer, steps, rate in (('pilot', 'sgd', 4, 1e-5), ('full', 'adamw', 158, 1e-4)):
                for method in ('sampled', 'exact'):
                    for seed in (0, 1):
                        name = f"{model['name']}_reward_{phase}_{method}_seed{seed}"
                        directory = output / 'experiments' / experiment_of(name) / 'training' / name
                        config = {'base': model['base'], 'adapter': str(initial_adapter),
                            'expected_base_hash': base_hash, 'expected_adapter_hash': initial['adapter_hash'],
                            'data': str(data / 'original'), 'output': str(directory), 'method': method,
                            'seed': seed, 'optimizer': optimizer, 'steps': 2 if args.smoke else steps,
                            'learning_rate': rate, 'device': args.device, 'dtype': args.dtype,
                            'group_size': 32, 'prefix_batch': args.prefix_batch}
                        path = output / 'configs' / (name + '.json')
                        configs[path] = config
                        dependencies = ['audit_data', 'reward_gradient_check', f"{model['name']}_previous_composition_seed0"]
                        if phase == 'full':
                            dependencies += pilots
                        add(name, 'iclr.reward_control', ['--config', path], directory, 'train', dependencies)
                        evaluate(name + '_closed', 'original', f'reward_{phase}_{method}', seed,
                                 directory / 'adapter', dependencies=(name,))
    by_id = {job['id']: job for job in jobs}
    selected = {job['id'] for job in jobs if job['resource'] == 'gpu' and job['experiment'] in experiments
                and (job['seed'] is None or job['seed'] in seeds)}
    if not selected:
        raise ValueError('No jobs match the requested experiments and seeds')
    pending = list(selected)
    while pending:
        for dependency in by_id[pending.pop()]['depends_on']:
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    jobs = [job for job in jobs if job['id'] in selected]
    referenced_paths = set()
    for job in jobs:
        referenced_paths.update(job['argv'])
        path = Path(job['argv'][-1])
        if path in configs:
            write_config(path, configs[path])
            referenced_paths.update(value for value in configs[path].values() if isinstance(value, str))
    models = [m for m in models if any(j['id'].startswith(m['name'] + '_') for j in jobs)]
    reference_hashes = set()
    for model in models:
        if model.get('receipt'):
            history = json.loads(Path(model['receipt']).read_text())
            reference_hashes.update(r['data_hash'] for r in history['runs'])
            model['required_adapters'] = [str(Path(model['previous']) / r['run'] / 'adapter')
                for r in history['runs'] if str(Path(model['previous']) / r['run'] / 'adapter') in referenced_paths]
    if len(reference_hashes) != 1:
        raise ValueError('Inherited models need the same verified reference dataset')
    queue = {'schema': 'iclr.upgrade.queue.v2', 'source_code_hash': code_hash(), 'smoke': args.smoke,
             'experiments': experiments, 'seeds': seeds, 'reference_hash': reference_hashes.pop(),
             'models': models, 'reference': args.reference, 'jobs': sorted(jobs, key=lambda j: j['priority']),
             'scope': {e: EXPERIMENTS[e] for e in experiments},
             'external_replication': 'SmolLM2-1.7B and matched fresh Qwen0.6 continuations' if include_external
             else 'G11 requires --include-external; long-horizon item in the source plan'}
    write_config(output / 'queue.json', queue)
    write_json(output / 'status.json', {'queue_sha256': file_hash(output / 'queue.json'), 'jobs': {
        job['id']: {'status': 'planned'} for job in jobs}}) if not (output / 'status.json').exists() else None
    print(json.dumps({'queue': str(output / 'queue.json'), 'jobs': len(jobs),
                      'training_jobs': sum(j['kind'] in ('train', 'init') for j in jobs), 'executed': False}))


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19], 'host': socket.gethostname(),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def verify_job(job):
    directory = Path(job['result'])
    if job['kind'] == 'data':
        protocol = json.loads((directory / 'protocol.json').read_text())
        for dataset, entry in protocol['datasets'].items():
            if file_hash(directory / dataset / 'manifest.json') != entry['manifest_sha256']:
                raise ValueError('Data protocol manifest changed')
            verify_data(directory / dataset)
        return file_hash(directory / 'protocol.json')
    receipt = verify_receipt(directory)
    if job['kind'] in ('train', 'init'):
        payload = directory / ('checkpoint' if job['kind'] == 'init' else 'adapter')
        if tree_hash(payload) != receipt['payload_hash']:
            raise ValueError('Trained payload hash changed')
    return file_hash(directory / 'DONE')


def run_queue(args):
    path = Path(args.queue)
    queue = json.loads(path.read_text())
    if queue['source_code_hash'] != code_hash():
        raise ValueError('Code changed after planning; prepare a new queue directory')
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError('Specify distinct GPU indices; one process will occupy each GPU')
    if args.cpu_threads < 1:
        raise ValueError('Positive CPU thread count required')
    if queue.get('reference_hash') and file_hash(Path(queue['reference']) / 'manifest.json') != queue['reference_hash']:
        raise ValueError('Reference dataset differs from the inherited training data')
    for model in queue['models']:
        if not model.get('receipt'):
            continue
        history = json.loads(Path(model['receipt']).read_text())
        if tree_hash(model['base']) != history['runs'][0]['base_hash']:
            raise ValueError(f"Wrong inherited atomic checkpoint: {model['base']}")
        for run in history['runs']:
            adapter = Path(model['previous']) / run['run'] / 'adapter'
            if 'required_adapters' in model and str(adapter) not in model['required_adapters']:
                continue
            if tree_hash(adapter) != run['adapter_hash']:
                raise ValueError(f'Wrong inherited continuation adapter: {adapter}')
    status_path = path.with_name('status.json')
    state = json.loads(status_path.read_text())
    if state['queue_sha256'] != file_hash(path):
        raise ValueError('Queue changed after status was recorded')
    lock = path.with_name('.queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = path.parent / 'logs'
    logs.mkdir(exist_ok=True)
    active = {}
    for job in queue['jobs']:
        item = state['jobs'][job['id']]
        if item['status'] == 'running':
            previous = item.get('process')
            if previous and previous['host'] != socket.gethostname():
                raise ValueError('A running job belongs to another host; inspect that process before resuming')
            if previous and identity(previous['pid']) == previous:
                active[job['id']] = (job, None, None, item['slot'])
                continue
            item['status'] = 'interrupted'
            try:
                item.update(status='done', receipt_sha256=verify_job(job))
            except (FileNotFoundError, ValueError):
                pass
        if item['status'] == 'done':
            if verify_job(job) != item['receipt_sha256']:
                raise ValueError('Completed job receipt changed')
        elif args.retry_failed and item['status'] in ('failed', 'interrupted', 'blocked_dependency'):
            item['status'] = 'planned'
    state['dispatcher'] = identity(os.getpid())
    write_json(status_path, state)
    try:
        while True:
            for name, (job, process, log, slot) in list(active.items()):
                item = state['jobs'][name]
                alive = process.poll() is None if process else identity(item['process']['pid']) == item['process']
                if alive:
                    continue
                if log:
                    log.close()
                try:
                    if process and process.returncode:
                        raise ValueError(f'Process exited {process.returncode}; see logs/{name}.log')
                    item.update(status='done', receipt_sha256=verify_job(job), ended_at=time.time())
                except Exception as error:
                    item.update(status='failed', error=str(error), ended_at=time.time())
                del active[name]
                print(name, item['status'], flush=True)
            occupied = {entry[3] for entry in active.values()}
            for job in queue['jobs']:
                item = state['jobs'][job['id']]
                if item['status'] != 'planned':
                    continue
                dependencies = [state['jobs'][name]['status'] for name in job['depends_on']]
                if any(s in ('failed', 'interrupted', 'blocked_dependency') for s in dependencies):
                    item['status'] = 'blocked_dependency'
                    continue
                if any(s != 'done' for s in dependencies):
                    continue
                slots = ['cpu'] if job['resource'] == 'cpu' else args.gpus
                free = [slot for slot in slots if slot not in occupied]
                if not free:
                    continue
                slot = free[0]
                env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': str(args.cpu_threads)}
                if slot != 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = slot
                log = (logs / (job['id'] + '.log')).open('a')
                process = subprocess.Popen([sys.executable, *job['argv']], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                item.update(status='running', process=identity(process.pid), slot=slot, started_at=time.time())
                active[job['id']] = (job, process, log, slot)
                occupied.add(slot)
                write_json(status_path, state)
                print('Started', job['id'], 'slot', slot, 'PID', process.pid, flush=True)
            write_json(status_path, state)
            if not active:
                break
            time.sleep(2)
    finally:
        # On dispatcher interruption, leave child process identities recorded.
        # A resume checks the actual PID/start time before scheduling anything.
        write_json(status_path, state)
        lock.close()
    if any(r['status'] != 'done' for r in state['jobs'].values()):
        raise RuntimeError('Queue has incomplete jobs; inspect status.json and logs; retry requires --retry-failed')
    if getattr(args, 'send_on_complete', False):
        send_results(argparse.Namespace(out=str(path.parent), destination=None, allow_incomplete=False))


def start(args):
    plan(args)
    args.queue = str(Path(args.out) / 'queue.json')
    run_queue(args)


def import_data(args):
    source, target = Path(args.source), Path(args.out)
    digest = verify_job({'kind': 'data', 'result': str(source)})
    protocol = json.loads((source / 'protocol.json').read_text())
    if args.expected_protocol_hash and digest != args.expected_protocol_hash:
        raise ValueError('Prepared data changed since planning')
    if file_hash(Path(args.reference) / 'manifest.json') != protocol['reference_manifest_sha256']:
        raise ValueError('Prepared and reference datasets do not match')
    if target.exists():
        if verify_job({'kind': 'data', 'result': str(target)}) != digest:
            raise ValueError('Destination data differ; use a new directory')
        return
    temporary = target.with_name(target.name + '.copying')
    shutil.copytree(source, temporary)
    if verify_job({'kind': 'data', 'result': str(temporary)}) != digest:
        raise ValueError('Copied data failed verification')
    temporary.rename(target)


def bundle(args):
    root, target = Path(args.out).resolve(), Path(args.archive).resolve()
    if target.is_relative_to(root) or target.exists():
        raise ValueError('Choose a new archive outside the result directory')
    state = json.loads((root / 'status.json').read_text())
    queue = json.loads((root / 'queue.json').read_text())
    if queue['source_code_hash'] != code_hash():
        raise ValueError('Export with the same code version used to prepare this queue')
    if any(r['status'] == 'running' for r in state['jobs'].values()):
        raise ValueError('Queue still records running jobs; refresh status via the dispatcher first')
    if not args.allow_incomplete and any(r['status'] != 'done' for r in state['jobs'].values()):
        raise ValueError('Incomplete queue; use --allow-incomplete only for an explicitly partial export')
    collect(argparse.Namespace(out=str(root)))
    files = [(p, 'results/' + p.relative_to(root).as_posix()) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != '.queue.lock' and
             (args.include_models or p.suffix not in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt'))]
    if queue.get('reference'):
        reference = Path(queue['reference'])
        manifest = verify_data(reference)
        if file_hash(reference / 'manifest.json') != queue['reference_hash']:
            raise ValueError('Inherited reference data changed before export')
        files.extend((reference / name, 'results/reference_data/' + name)
                     for name in ['manifest.json', *manifest['files']])
    source = [*(ROOT / 'iclr').glob('*.py'), *(ROOT / 'legacy').glob('*.py'),
              ROOT / 'uv.lock', ROOT / 'pyproject.toml', ROOT / 'README.md',
              *(ROOT / 'plans').glob('*.json'), ROOT / 'plans/GOAL_TO_ICLR.md',
              ROOT / 'plans/source/goal_to_iclr.md', *(ROOT / 'evidence/upgrade').rglob('*')]
    source = [p for p in source if p.is_file()]
    files.extend((p, 'code/' + p.relative_to(ROOT).as_posix()) for p in source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.partial')
    manifest = []
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for path, name in files:
            digest = file_hash(path)
            archive.write(path, name)
            if file_hash(path) != digest:
                raise ValueError(f'File changed during export: {path}')
            manifest.append({'path': name, 'sha256': digest, 'bytes': path.stat().st_size})
        guide = ('# Результаты для Артёма\n\n'
            'Начните с results/analysis/summary.csv и results/analysis/runs.csv.\n'
            'В results/experiments/ лежат оценки и обучение по смысловым блокам.\n'
            'Сырые ответы, score, префиксы, calibration, данные, конфиги и логи включены.\n'
            'Веса моделей ' + ('включены.' if args.include_models else 'исключены; для анализа они не нужны.') + '\n'
            'Полнота и SHA-256 каждого файла: EXPORT_MANIFEST.json.\n'
            'Не усредняйте разные data_hash, evaluation_set и target_control.\n\n' +
            '\n'.join(f'- {name}: {EXPERIMENTS[name]}' for name in queue.get('experiments', [])) + '\n')
        payload = guide.encode()
        archive.writestr('START_HERE_RU.md', payload)
        manifest.append({'path': 'START_HERE_RU.md', 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
        archive.writestr('EXPORT_MANIFEST.json', json.dumps({'includes_weights': args.include_models,
            'status': 'complete' if all(r['status'] == 'done' for r in state['jobs'].values()) else 'partial',
            'source_code_hash': code_hash(), 'files': manifest}, indent=2))
    verify_bundle(argparse.Namespace(archive=str(temporary)))
    temporary.replace(target)
    print(json.dumps({'archive': str(target), 'sha256': file_hash(target), 'files': len(manifest)}))


def send_results(args):
    root = Path(args.out).resolve()
    folder = Path(args.destination).resolve() if args.destination else root.parent / (
        'send_to_artem_exp_' + root.name + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    if folder.exists() or folder.is_relative_to(root):
        raise ValueError('Choose a new send_to_artem folder outside the experiment working directory')
    archive = folder.with_name(folder.name + '.zip')
    bundle(argparse.Namespace(out=str(root), archive=str(archive), include_models=False,
                              allow_incomplete=args.allow_incomplete))
    with zipfile.ZipFile(archive) as saved:
        if any(not (folder / name).resolve().is_relative_to(folder) for name in saved.namelist()):
            raise ValueError('Archive path escapes the send folder')
        saved.extractall(folder)
    print(f'ОТПРАВИТЬ АРТЁМУ: {archive}\nПапка с тем же содержимым: {folder}', flush=True)


def collect(args):
    """Index existing results with portable paths; never pool different datasets or seeds."""
    root = Path(args.out).resolve()
    state = json.loads((root / 'status.json').read_text())
    queue_path = root / 'queue.json'
    if state['queue_sha256'] != file_hash(queue_path):
        raise ValueError('Queue changed after status was recorded')
    index, summaries = [], []
    for job in json.loads(queue_path.read_text())['jobs']:
        directory = Path(job['result']).resolve()
        row = {'job': job['id'], 'kind': job['kind'], 'status': state['jobs'][job['id']]['status'],
               'experiment': job.get('experiment'), 'seed': job.get('seed'),
               'result': directory.relative_to(root).as_posix()}
        if row['status'] == 'done':
            row['receipt_sha256'] = verify_job(job)
            if row['receipt_sha256'] != state['jobs'][job['id']]['receipt_sha256']:
                raise ValueError('Completed job receipt changed')
            if (directory / 'summary.csv').is_file():
                binding = json.loads((directory / 'binding.json').read_text())
                with (directory / 'summary.csv').open() as stream:
                    summaries.extend({**item, 'job': job['id'], 'result': row['result'],
                        'experiment': job.get('experiment'),
                        **{key: binding[key] for key in ('data_hash', 'base_hash', 'adapter_hash')}}
                        for item in csv.DictReader(stream))
        index.append(row)
    for name, rows in (('runs.csv', index), ('summary.csv', summaries)):
        path = root / 'analysis' / name
        path.parent.mkdir(exist_ok=True)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'index': str(root / 'analysis'), 'jobs': len(index), 'summary_rows': len(summaries)}))


def verify_bundle(args):
    """Check a downloaded archive without original outputs or model weights."""
    with zipfile.ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read('EXPORT_MANIFEST.json'))
        names = [row['path'] for row in manifest['files']]
        if (len(set(names)) != len(names) or len(set(archive.namelist())) != len(archive.namelist()) or
                set(archive.namelist()) != {*names, 'EXPORT_MANIFEST.json'}):
            raise ValueError('Archive inventory differs from the manifest')
        for row in manifest['files']:
            digest, size = hashlib.sha256(), 0
            with archive.open(row['path']) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != row['sha256'] or size != row['bytes']:
                raise ValueError(f"Archive content failed verification: {row['path']}")
    print(json.dumps({'archive': str(args.archive), 'verified_files': len(names), 'status': manifest['status']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, function in (('plan', plan), ('start', start)):
        p = sub.add_parser(name)
        p.add_argument('--models', default='plans/upgrade_models.json')
        p.add_argument('--inputs', help='shared folder containing reference_data and new_tasks')
        p.add_argument('--reference', default='outputs/data')
        p.add_argument('--prepared-data', help='shared frozen data copied identically to each computer')
        p.add_argument('--experiments', nargs='+', choices=EXPERIMENTS)
        p.add_argument('--seeds', nargs='+', type=int, choices=(0, 1, 2), default=[0, 1, 2])
        p.add_argument('--out', required=True)
        p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
        p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
        p.add_argument('--prefix-batch', type=int, default=32)
        p.add_argument('--smoke', action='store_true')
        p.add_argument('--include-grpo', action='store_true', help='add exact-gradient check, paired SGD pilots and matched AdamW runs')
        p.add_argument('--include-external', action='store_true', help='add pinned SmolLM2-1.7B with matched fresh Qwen0.6 continuations')
        if name == 'start':
            p.add_argument('--gpus', nargs='+', required=True)
            p.add_argument('--cpu-threads', type=int, default=4)
            p.add_argument('--retry-failed', action='store_true')
        p.set_defaults(function=function, send_on_complete=True)
    r = sub.add_parser('run')
    r.add_argument('--queue', required=True)
    r.add_argument('--gpus', nargs='+', required=True)
    r.add_argument('--cpu-threads', type=int, default=4)
    r.add_argument('--retry-failed', action='store_true')
    r.set_defaults(function=run_queue, send_on_complete=True)
    s = sub.add_parser('send', help='create send_to_artem_exp folder and ZIP without model weights')
    s.add_argument('--out', required=True)
    s.add_argument('--destination')
    s.add_argument('--allow-incomplete', action='store_true')
    s.set_defaults(function=send_results)
    i = sub.add_parser('import-data')
    i.add_argument('--source', required=True)
    i.add_argument('--reference', required=True)
    i.add_argument('--out', required=True)
    i.add_argument('--expected-protocol-hash')
    i.set_defaults(function=import_data)
    b = sub.add_parser('bundle')
    b.add_argument('--out', required=True)
    b.add_argument('--archive', required=True)
    b.add_argument('--include-models', action='store_true')
    b.add_argument('--allow-incomplete', action='store_true')
    b.set_defaults(function=bundle)
    c = sub.add_parser('collect', help='collect verified per-seed summaries and a portable run index')
    c.add_argument('--out', required=True)
    c.set_defaults(function=collect)
    v = sub.add_parser('verify-bundle', help='verify SHA-256 and inventory after downloading, without weights')
    v.add_argument('--archive', required=True)
    v.set_defaults(function=verify_bundle)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()

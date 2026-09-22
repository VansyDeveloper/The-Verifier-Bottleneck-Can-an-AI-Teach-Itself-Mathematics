"""Input and receipt checks shared by v4 workers."""

import json
from pathlib import Path

from .common import ROOT, code_hash, environment, file_hash, tree_hash, verify_data, verify_receipt, write_json


def plan_path(version='v4'):
    if version not in ('v4', 'v5'):
        raise ValueError('Unknown research plan; use v4 or v5 explicitly')
    return ROOT / 'plans' / f'research_{version}.json'


def exposure_ledger(manifest, stages):
    motifs = ['>'.join(p) for p in manifest['constraints'].get('heldout_motifs', [])]
    known = bool(stages) and all(s.get('provenance_status') == 'verified' and s.get('pairs') is not None for s in stages)
    depth_known = bool(stages) and all(s.get('provenance_status') == 'verified' and s.get('depths')
        and all(n is not None for n in s['depths'].values()) for s in stages)
    status = ('verified' if known else 'unknown' if any(s.get('pairs') is None for s in stages) else 'declared_only')
    return {'stages': stages, 'mask_pairs': motifs,
        'provenance_status': status, 'depth_history_status': 'verified' if depth_known else 'unknown',
        'recorded_history_excludes_mask': known and all(not s['pairs'].get(p, 0) for s in stages for p in motifs),
        'historical_depth4': any(int(d) >= 4 and n for s in stages for d, n in s.get('depths', {}).items()),
        'scope': 'witness/correct-solution supervision in recorded task-training stages; base pretraining unknown; complete-space scoring visits all legal prefixes'}


def start_run(config, kind):
    data, out = Path(config['data']), Path(config['output'])
    manifest = verify_data(data)
    if manifest['schema'] != 'iclr.research.data.v4':
        raise ValueError('Expected a frozen v4 dataset')
    binding = {'kind': kind, 'config': config, 'data_hash': file_hash(data / 'manifest.json'),
               'code_hash': code_hash(), 'plan': config.get('plan', 'v4'),
               'plan_hash': file_hash(plan_path(config.get('plan', 'v4'))),
               'base_hash': tree_hash(config['base']),
               'adapter_hash': tree_hash(config['adapter']) if config.get('adapter') else None}
    if config.get('analysis_lock'):
        binding['analysis_lock_hash'] = file_hash(config['analysis_lock'])
    if config.get('gate'):
        binding['gate_receipt_hash'] = file_hash(Path(config['gate']) / 'DONE')
    if config.get('atomic_gate'):
        binding['atomic_gate_receipt_hash'] = file_hash(Path(config['atomic_gate']) / 'DONE')
    initial = manifest.get('historical_exposure', {}).get(binding['adapter_hash'])
    stages = [{'stage': 'declared_atomic_task_initialization', 'provenance_status': 'declared_only',
               'pairs': {}, 'depths': {}, 'scope': 'role declaration only; task-training history not verified'}]
    if config.get('base_training_receipt'):
        path = Path(config['base_training_receipt'])
        prior = verify_receipt(path)
        if prior.get('payload_hash') != binding['base_hash']:
            raise ValueError('Base training receipt differs from the loaded base')
        stages = json.loads((path / 'exposure.json').read_text())['stages']
        binding['base_training_receipt_hash'] = file_hash(path / 'DONE')
    if binding['adapter_hash']:
        stages.append(initial or {'stage': 'unrecorded_adapter_history', 'provenance_status': 'unknown', 'pairs': None, 'depths': {}})
    if config.get('training_receipt'):
        trained = verify_receipt(config['training_receipt'])
        prior = trained['binding']
        if (trained['payload_hash'] != binding['adapter_hash'] or prior['base_hash'] != binding['base_hash']
                or prior['data_hash'] != binding['data_hash'] or prior['code_hash'] != binding['code_hash']
                or prior['plan_hash'] != binding['plan_hash']
                or prior['config']['seed'] != config['seed'] or prior['config']['objective'] != config['arm']):
            raise ValueError('Evaluation differs from its training receipt')
        binding['training_receipt_hash'] = file_hash(Path(config['training_receipt']) / 'DONE')
        binding['initial_adapter_hash'] = prior['adapter_hash']
        binding['checkpoint_step'] = trained.get('checkpoint_step')
        stages = json.loads((Path(config['training_receipt']) / 'exposure.json').read_text())['stages']
    if config.get('initial_training_receipt'):
        path = Path(config['initial_training_receipt'])
        initialized = verify_receipt(path)
        if (initialized['payload_hash'] != binding.get('initial_adapter_hash', binding['adapter_hash'])
                or initialized['binding']['base_hash'] != binding['base_hash']):
            raise ValueError('Initial training receipt differs from the loaded adapter/base')
        binding['initial_training_receipt_hash'] = file_hash(path / 'DONE')
        if not config.get('training_receipt'):
            stages = json.loads((path / 'exposure.json').read_text())['stages']
    binding['exposure'] = exposure_ledger(manifest, stages)
    if config.get('amendment'):
        from .research_data import verify_amendment
        verify_amendment(config['amendment'], data, binding['plan_hash'])
        binding['amendment_hash'] = file_hash(config['amendment'])
    if kind == 'reward_diagnostic' and not binding['exposure']['recorded_history_excludes_mask']:
        raise ValueError('Reward initialization history does not exclude this mask; use original or matching-mask SFT')
    if config.get('atomic_gate'):
        require_gate(config['atomic_gate'], 'atomic', {**binding,
            'adapter_hash': binding.get('initial_adapter_hash', binding['adapter_hash'])})
    for field in ('base_hash', 'adapter_hash'):
        if config.get('expected_' + field) and config['expected_' + field] != binding[field]:
            raise ValueError('Checkpoint differs from the declared initialization')
    if (out / 'DONE').exists():
        receipt = verify_receipt(out)
        if receipt['binding'] != binding:
            raise ValueError('Completed output has different inputs')
        if receipt.get('payload_hash') and tree_hash(out / 'adapter') != receipt['payload_hash']:
            raise ValueError('Trained adapter changed')
        return manifest, binding, True
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'binding.json').exists() and json.loads((out / 'binding.json').read_text()) != binding:
        raise ValueError('Partial output belongs to another configuration; choose a new directory')
    write_json(out / 'binding.json', binding)
    write_json(out / 'environment.json', environment())
    return manifest, binding, False


def finish_run(out, binding, payload=False):
    out = Path(out)
    receipt = {'binding': binding, 'files': {str(p.relative_to(out)): file_hash(p)
        for p in out.rglob('*') if p.is_file() and 'adapter' not in p.relative_to(out).parts
        and p != out / 'DONE' and p.name not in ('FAILED', 'resume_state.pt')}}
    if payload:
        receipt['payload_hash'] = tree_hash(out / 'adapter')
    write_json(out / 'DONE', receipt)
    (out / 'FAILED').unlink(missing_ok=True)


def require_gate(path, kind, binding):
    if not path:
        raise ValueError(f'{kind} requires its completed dev diagnostic gate')
    receipt = verify_receipt(path)
    prior = receipt['binding']
    gate = json.loads((Path(path) / 'gate.json').read_text())
    if gate.get('kind') != kind or gate.get('pass') is not True:
        raise ValueError(f'{kind} diagnostic gate did not pass: {gate.get("status", "insufficient dev evidence")}')
    for field in ('code_hash', 'plan_hash', 'data_hash', 'base_hash', 'adapter_hash'):
        if prior[field] != binding[field]:
            raise ValueError(f'{kind} gate belongs to another {field}')
    return file_hash(Path(path) / 'DONE')

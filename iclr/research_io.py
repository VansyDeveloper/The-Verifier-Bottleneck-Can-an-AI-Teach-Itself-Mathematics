"""Input and receipt checks shared by v3 workers."""

import json
from pathlib import Path

from .common import ROOT, code_hash, environment, file_hash, tree_hash, verify_data, verify_receipt, write_json


def start_run(config, kind):
    data, out = Path(config['data']), Path(config['output'])
    manifest = verify_data(data)
    if manifest['schema'] != 'iclr.research.data.v3':
        raise ValueError('Expected a frozen v3 dataset')
    binding = {'kind': kind, 'config': config, 'data_hash': file_hash(data / 'manifest.json'),
               'code_hash': code_hash(), 'plan_hash': file_hash(ROOT / 'plans/research_v3.json'),
               'base_hash': tree_hash(config['base']),
               'adapter_hash': tree_hash(config['adapter']) if config.get('adapter') else None}
    if config.get('analysis_lock'):
        binding['analysis_lock_hash'] = file_hash(config['analysis_lock'])
    if config.get('gate'):
        binding['gate_receipt_hash'] = file_hash(Path(config['gate']) / 'DONE')
    if config.get('training_receipt'):
        trained = verify_receipt(config['training_receipt'])
        prior = trained['binding']
        if (trained['payload_hash'] != binding['adapter_hash'] or prior['base_hash'] != binding['base_hash']
                or prior['data_hash'] != binding['data_hash'] or prior['code_hash'] != binding['code_hash']
                or prior['config']['seed'] != config['seed'] or prior['config']['objective'] != config['arm']):
            raise ValueError('Evaluation differs from its training receipt')
        binding['training_receipt_hash'] = file_hash(Path(config['training_receipt']) / 'DONE')
        binding['initial_adapter_hash'] = prior['adapter_hash']
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
        and p.name not in ('DONE', 'FAILED')}}
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
        raise ValueError(f'{kind} diagnostic gate did not pass')
    for field in ('code_hash', 'plan_hash', 'data_hash', 'base_hash', 'adapter_hash'):
        if prior[field] != binding[field]:
            raise ValueError(f'{kind} gate belongs to another {field}')
    return file_hash(Path(path) / 'DONE')

"""No-model reproductions and proposed recovery patch checks.
Runs exact source AST for configuration handling. Does not load an LLM or validate
Qwen/LoRA/Adam continuation; those need the project's full regression environment.
"""
import argparse, ast, difflib, hashlib, json, math, tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True, help='Unmodified 056fd9a source tree')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
ROOT = args.repo.resolve()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
source_path = ROOT / 'iclr/research_train.py'
original = source_path.read_text()

def make_config_runner(source):
    tree = ast.parse(source)
    chosen = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'OBJECTIVES','DEFAULTS'} for t in node.targets):
            chosen.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {'config_checked','run'}:
            chosen.append(node)
    ns = {'math': math, 'start_run': lambda cfg, kind: ({}, {}, True)}
    exec(compile(ast.Module(body=chosen,type_ignores=[]), str(source_path), 'exec'), ns)
    return ns['run']

checks = {}
run = make_config_runner(original)
run({'resume_from':'latest'})
checks['original_config_only_resume'] = 'passes configuration parsing (start_run returns completed sentinel)'
try:
    run({'resume_from':'latest'}, resume_from='latest')
except ValueError as exc:
    checks['original_explicit_plus_config_resume'] = str(exc)
else:
    raise AssertionError('Expected original explicit/config conflict')

# The original resume block parses every record before discarding future steps.
# Use the actual unchanged read_jsonl function, with one valid committed record
# followed by an unfinished uncommitted line.
import sys
sys.path.insert(0, str(ROOT))
from iclr.common import read_jsonl
with tempfile.TemporaryDirectory() as temp:
    p = Path(temp) / 'training_metrics.jsonl'
    p.write_bytes(b'{"step": 1, "loss": 0.4}\n{"step": 2, "loss":')
    try:
        [r for r in read_jsonl(p) if r['step'] <= 1]
    except json.JSONDecodeError as exc:
        checks['original_torn_tail_recovery'] = type(exc).__name__
    else:
        raise AssertionError('Expected original recovery parsing failure')

helpers = '''

LOG_NAMES = ('training_metrics.jsonl', 'training_stream.jsonl', 'replay_stream.jsonl')


def snapshot_logs(out):
    """Record committed byte boundaries; called after all training logs are flushed."""
    result = {}
    for name in LOG_NAMES:
        path = Path(out) / name
        contents = path.read_bytes() if path.exists() else b''
        result[name] = {'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}
    return result


def restore_logs(out, snapshot):
    """Verify every committed prefix before discarding uncommitted trailing bytes."""
    if not isinstance(snapshot, dict) or set(snapshot) != set(LOG_NAMES):
        raise ValueError('Checkpoint has no complete committed-log snapshot; do not guess recovery boundaries')
    # Validate all files first: never silently discard damage inside a checkpoint.
    for name in LOG_NAMES:
        entry = snapshot[name]
        size = entry.get('bytes')
        if type(size) is not int or size < 0:
            raise ValueError(f'Invalid committed log boundary: {name}')
        path = Path(out) / name
        if path.exists():
            with path.open('rb') as stream:
                prefix = stream.read(size)
        else:
            prefix = b''
        if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != entry.get('sha256'):
            raise ValueError(f'Committed log prefix is missing or damaged: {name}')
    for name in LOG_NAMES:
        path = Path(out) / name
        if path.exists():
            with path.open('r+b') as stream:
                stream.truncate(snapshot[name]['bytes'])


def complete_checkpoint_receipts(out):
    """A .partial directory is not an atomically published checkpoint."""
    return sorted(p for p in (Path(out) / 'checkpoints').glob('step_*/DONE')
                  if p.parent.name.removeprefix('step_').isdigit())
'''
patched = original.replace('import argparse\n', 'import argparse\nimport hashlib\n',1)
patched = patched.replace('\n\ndef checkpoint(', helpers + '\n\ndef checkpoint(',1)
patched = patched.replace("'counts': dict(counts)}, temporary / 'resume_state.pt')", "'counts': dict(counts), 'logs': snapshot_logs(out)}, temporary / 'resume_state.pt')",1)
patched = patched.replace("    resume_from = resume_from or config.pop('resume_from', None)", "    configured_resume = config.pop('resume_from', None)\n    resume_from = resume_from if resume_from is not None else configured_resume",1)
patched = patched.replace("sorted((out / 'checkpoints').glob('step_*/DONE'))", 'complete_checkpoint_receipts(out)')
old = '''        for name, keep in [('training_metrics.jsonl', lambda r: r['step'] <= first_step),
                           ('training_stream.jsonl', lambda r: r['step'] < first_step),
                           ('replay_stream.jsonl', lambda r: r['step'] <= first_step)]:
            path = out / name
            if path.exists():
                path.write_bytes(core.canonical_jsonl_bytes([r for r in read_jsonl(path) if keep(r)]))'''
assert old in patched
patched = patched.replace(old, "        restore_logs(out, state.get('logs'))",1)
compile(patched,str(source_path),'exec')
run = make_config_runner(patched)
run({'resume_from':'latest'},resume_from='latest')
checks['patched_explicit_plus_config_resume'] = 'PASS'

ns = {'Path': Path, 'hashlib': hashlib}
exec(helpers,ns)
with tempfile.TemporaryDirectory() as temp:
    out = Path(temp)
    originals = {}
    for i,name in enumerate(ns['LOG_NAMES']):
        b = json.dumps({'step':1,'id':i}).encode() + b'\n'
        (out/name).write_bytes(b); originals[name]=b
    snap = ns['snapshot_logs'](out)
    for name in ns['LOG_NAMES']:
        with (out/name).open('ab') as f:f.write(b'{"step":2,"loss":')
    ns['restore_logs'](out,snap)
    assert all((out/name).read_bytes()==originals[name] for name in originals)
    checks['patched_torn_tail_recovery'] = 'PASS; committed bytes preserved exactly'
    # Corruption of committed data must fail before changing any other file.
    bad = out/'training_stream.jsonl'; bad.write_bytes(b'X'+originals[bad.name][1:])
    before = {name:(out/name).read_bytes() for name in originals}
    try:ns['restore_logs'](out,snap)
    except ValueError:pass
    else:raise AssertionError('Committed corruption must fail')
    assert before == {name:(out/name).read_bytes() for name in originals}
    checks['patched_rejects_committed_corruption_without_mutation'] = 'PASS'
    for name in ['step_000001','step_000002.partial']:
        cp=out/'checkpoints'/name;cp.mkdir(parents=True);(cp/'DONE').write_text('{}')
    assert [p.parent.name for p in ns['complete_checkpoint_receipts'](out)] == ['step_000001']
    checks['patched_ignores_unpublished_partial_checkpoint'] = 'PASS'
with tempfile.TemporaryDirectory() as temp:
    out=Path(temp);snap=ns['snapshot_logs'](out)
    (out/'training_metrics.jsonl').write_bytes(b'uncommitted')
    ns['restore_logs'](out,snap)
    assert (out/'training_metrics.jsonl').read_bytes()==b''
    checks['patched_empty_step0_logs'] = 'PASS'

patch=''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='a/iclr/research_train.py',tofile='b/iclr/research_train.py'))
(OUT/'resume_recovery.patch').write_text(patch)
(OUT/'proposed_research_train.py').write_text(patched)
(OUT/'resume_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(checks,indent=2,ensure_ascii=False))

import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_receipt(directory, name='DONE'):
    directory = Path(directory).resolve()
    receipt = json.loads((directory / name).read_text())
    files = receipt.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError(f'Completion receipt has no file hashes: {directory}')
    for name, digest in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f'Receipt path leaves the run directory: {name}')
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Completed output missing or changed: {path}')
    return receipt


def tree_hash(path):
    digest = hashlib.sha256()
    files = sorted(p for p in Path(path).rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'Empty payload: {path}')
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode() + b'\0')
        digest.update(bytes.fromhex(file_hash(item)))
    return digest.hexdigest()


def code_hash():
    digest = hashlib.sha256()
    for item in sorted([*(ROOT / 'iclr').glob('*.py'), *(ROOT / 'legacy').glob('*.py')]):
        digest.update(str(item.relative_to(ROOT)).encode() + b'\0')
        digest.update(bytes.fromhex(file_hash(item)))
    return digest.hexdigest()


def environment():
    import torch
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, status = None, 'unavailable: source archive without Git metadata'
    return {
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'peft', 'numpy', 'scipy')},
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'git_commit': commit,
        'git_status': status,
    }


def verify_data(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        if file_hash(path / name).lower() != entry['sha256'].lower():
            raise ValueError(f'Dataset hash mismatch: {name}')
    return manifest

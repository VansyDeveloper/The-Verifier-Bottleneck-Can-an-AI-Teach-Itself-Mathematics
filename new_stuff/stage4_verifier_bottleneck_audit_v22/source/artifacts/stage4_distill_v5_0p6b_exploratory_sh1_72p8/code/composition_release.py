from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
OPS = ("SH1", "SC2", "REV", "AC1", "AX1")
HIT_CUTOFFS = (1, 8, 16, 32, 64)
METRIC_NAMES = (*(f"hit@{cutoff}" for cutoff in HIT_CUTOFFS), "correct_mass", "best_rank", "mrr", "log_gap")
EXPECTED_CANDIDATES = {2: 25, 3: 125, 4: 625}
CONFIRM_SEEDS = tuple(range(6))
CONFIRM_BRANCHES = ("atomic_base", "atomic_control", "composition_distill")
CONFIRM_FROZEN_SPLITS = tuple(
    [f"final_{family}" for family in "abcd"]
    + [f"final_{family}_depth2" for family in "abcd"]
    + [f"final_{family}_depth4" for family in "abcd"]
)
_CACHE_PARTS = {"__pycache__", ".pytest_cache", ".cache", ".mypy_cache", ".ruff_cache"}
ATOMIC_PAYLOAD_MANIFEST = PurePosixPath("manifests/ATOMIC_PAYLOAD_SHA256SUMS.json")
_CRITICAL_ROOT_PARTS = {
    "adapters", "atomic_export", "configs", "data", "manifests", "rankings",
    "reports", "runs", "summary", "summaries",
}
_CRITICAL_ROOT_FILES = {"CONFIG_FROZEN.json", "CONFIG_SELECTION_FROZEN.json"}
_RECEIPT_PATHS = {
    PurePosixPath("manifests/archive_receipt.json"),
    PurePosixPath("archives/RELEASE.json"),
}


class ReleaseValidationError(RuntimeError):
    """Raised when a release input is incomplete, inconsistent, or corrupt."""


def _fail(message: str) -> NoReturn:
    raise ReleaseValidationError(message)


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            _fail(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def loads_strict(text: str, source: str = "<memory>") -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ReleaseValidationError:
        raise
    except Exception as exc:
        raise ReleaseValidationError(f"invalid JSON in {source}: {exc}") from exc


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        _fail(f"JSON path is not a regular file: {path}")
    try:
        return loads_strict(path.read_text(encoding="utf-8"), str(path))
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(f"JSON is not UTF-8: {path}: {exc}") from exc


def _is_jsonl(path: Path) -> bool:
    return path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz")


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        _fail(f"JSONL path is not a regular file: {path}")
    if not _is_jsonl(path):
        _fail(f"expected .jsonl or .jsonl.gz: {path}")
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            seen = 0
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    _fail(f"blank JSONL record at {path}:{line_number}")
                seen += 1
                yield line_number, loads_strict(line, f"{path}:{line_number}")
            if seen == 0:
                _fail(f"empty JSONL file: {path}")
    except (OSError, EOFError) as exc:
        raise ReleaseValidationError(f"unreadable or corrupt JSONL stream {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[Any]:
    return [value for _, value in iter_jsonl(path)]


def _jsonschema_module():
    try:
        import jsonschema
    except Exception as exc:  # pragma: no cover - exercised only in an incomplete environment
        raise ReleaseValidationError("jsonschema is required for release validation") from exc
    return jsonschema


def load_schema_registry(schema_dir: Path = SCHEMAS) -> dict[str, dict[str, Any]]:
    schema_dir = Path(schema_dir)
    registry: dict[str, dict[str, Any]] = {}
    jsonschema = _jsonschema_module()
    for path in sorted(schema_dir.glob("*.schema.json")):
        schema = load_json(path)
        if not isinstance(schema, dict):
            _fail(f"schema is not an object: {path}")
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except Exception as exc:
            raise ReleaseValidationError(f"invalid JSON Schema {path}: {exc}") from exc
        identifiers = [schema.get("$id"), *schema.get("x-schema-aliases", [])]
        for identifier in identifiers:
            if not isinstance(identifier, str) or not identifier:
                continue
            if identifier in registry:
                _fail(f"duplicate schema identifier {identifier!r} in {path}")
            registry[identifier] = schema
    if not registry:
        _fail(f"no schemas found in {schema_dir}")
    return registry


def validate_value(value: Any, schema: Mapping[str, Any], source: str) -> None:
    jsonschema = _jsonschema_module()
    try:
        jsonschema.Draft202012Validator(dict(schema)).validate(value)
    except Exception as exc:
        raise ReleaseValidationError(f"schema validation failed for {source}: {exc}") from exc


def _validate_declared(
    value: Any,
    registry: Mapping[str, Mapping[str, Any]],
    source: str,
    *,
    unknown_is_error: bool = False,
) -> bool:
    if not isinstance(value, dict):
        return False
    identifier = value.get("schema")
    if identifier is None:
        return False
    if not isinstance(identifier, str) or not identifier:
        if unknown_is_error:
            _fail(f"invalid schema id in critical artifact {source}: {identifier!r}")
        return False
    if identifier not in registry:
        if unknown_is_error:
            _fail(f"unknown schema id in critical artifact {source}: {identifier!r}")
        return False
    validate_value(value, registry[identifier], source)
    return True


def _critical_schema_path(rel: Path) -> bool:
    parts = rel.parts
    return bool(parts) and (parts[0] in _CRITICAL_ROOT_PARTS or rel.as_posix() in _CRITICAL_ROOT_FILES)


def validate_json_file(path: Path, schema_path: Path | None = None) -> Any:
    value = load_json(path)
    if schema_path is not None:
        validate_value(value, load_json(schema_path), str(path))
    return value


def validate_jsonl_file(path: Path, schema_path: Path | None = None) -> list[Any]:
    schema = load_json(schema_path) if schema_path is not None else None
    rows = []
    for line_number, value in iter_jsonl(path):
        if schema is not None:
            validate_value(value, schema, f"{path}:{line_number}")
        rows.append(value)
    return rows


def validate_tree(root: Path, schema_dir: Path | None = None) -> dict[str, Any]:
    """Parse every JSON stream and schema-check every record with a known schema id."""
    root = Path(root).resolve()
    registry = load_schema_registry(schema_dir or root / "schemas")
    result = {
        "json_files": 0,
        "jsonl_files": 0,
        "jsonl_rows": 0,
        "schema_validated_documents": 0,
        "schema_validated_rows": 0,
        "syntax_only_documents": 0,
        "syntax_only_rows": 0,
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if any(part in _CACHE_PARTS for part in rel.parts) or rel.parts[0] == "archives":
            continue
        if path.name.endswith(".schema.json") and rel.parts[0] == "schemas":
            load_json(path)
            result["json_files"] += 1
            continue
        if path.suffix == ".json":
            value = load_json(path)
            result["json_files"] += 1
            key = ("schema_validated_documents" if _validate_declared(
                value, registry, str(path), unknown_is_error=_critical_schema_path(rel)
            ) else "syntax_only_documents")
            result[key] += 1
        elif _is_jsonl(path):
            result["jsonl_files"] += 1
            for line_number, value in iter_jsonl(path):
                result["jsonl_rows"] += 1
                key = ("schema_validated_rows" if _validate_declared(
                    value, registry, f"{path}:{line_number}", unknown_is_error=_critical_schema_path(rel)
                ) else "syntax_only_rows")
                result[key] += 1
    return result


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        _fail(f"cannot hash non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseValidationError(f"cannot serialize strict JSON for {path}: {exc}") from exc
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def apply_operation(state: Sequence[int], operation: str, field: int) -> tuple[int, ...]:
    if not isinstance(field, int) or isinstance(field, bool) or field < 2:
        _fail(f"invalid field: {field!r}")
    current = tuple(int(value) % field for value in state)
    if operation == "SH1":
        out = [0] * len(current)
        for i, coefficient in enumerate(current):
            for j in range(i + 1):
                out[j] = (out[j] + coefficient * math.comb(i, j)) % field
        return tuple(out)
    if operation == "SC2":
        return tuple(coefficient * pow(2, i, field) % field for i, coefficient in enumerate(current))
    if operation == "REV":
        return tuple(reversed(current))
    out = list(current)
    if operation == "AC1":
        if not out:
            _fail("AC1 cannot be applied to an empty state")
        out[0] = (out[0] + 1) % field
    elif operation == "AX1":
        if len(out) < 2:
            _fail("AX1 requires at least two coefficients")
        out[1] = (out[1] + 1) % field
    else:
        _fail(f"unknown operation: {operation!r}")
    return tuple(out)


def apply_program(state: Sequence[int], program: Sequence[str], field: int) -> tuple[int, ...]:
    current = tuple(state)
    for operation in program:
        current = apply_operation(current, operation, field)
    return current


def enumerate_programs(depth: int) -> list[tuple[str, ...]]:
    if depth not in EXPECTED_CANDIDATES:
        _fail(f"ranking depth must be one of {sorted(EXPECTED_CANDIDATES)}; got {depth}")
    return list(itertools.product(OPS, repeat=depth))


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} is non-finite: {value!r}")
    return result


def recompute_metrics(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not entries:
        _fail("ranking is empty")
    scores = [_finite_number(entry.get("score"), f"ranking[{i}].score") for i, entry in enumerate(entries)]
    correct = [entry.get("correct") for entry in entries]
    if any(type(value) is not bool for value in correct):
        _fail("every ranking correct label must be boolean")
    correct_ranks = [index + 1 for index, value in enumerate(correct) if value]
    incorrect_scores = [score for score, value in zip(scores, correct) if not value]
    if not correct_ranks:
        _fail("full ranking has no exact-correct program")
    if not incorrect_scores:
        _fail("full ranking has no incorrect program; log-gap is undefined")
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(weights)
    if denominator <= 0 or not math.isfinite(denominator):
        _fail("invalid softmax normalizer")
    correct_mass = math.fsum(weight for weight, value in zip(weights, correct) if value) / denominator
    best_rank = min(correct_ranks)
    best_correct_score = max(score for score, value in zip(scores, correct) if value)
    best_incorrect_score = max(incorrect_scores)
    metrics: dict[str, Any] = {f"hit@{cutoff}": float(best_rank <= cutoff) for cutoff in HIT_CUTOFFS}
    metrics.update({
        "correct_mass": correct_mass,
        "best_rank": best_rank,
        "mrr": 1.0 / best_rank,
        "log_gap": best_correct_score - best_incorrect_score,
    })
    return metrics


def _metric_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: record[name] for name in METRIC_NAMES if name in record}


def assert_metrics_match(actual: Mapping[str, Any], expected: Mapping[str, Any], source: str, tolerance: float = 1e-12) -> None:
    for name, value in expected.items():
        if name not in actual:
            _fail(f"missing recomputed metric {name} for {source}")
        if name == "best_rank":
            if type(value) is not int or value != actual[name]:
                _fail(f"metric mismatch for {source}/{name}: reported={value!r}, recomputed={actual[name]!r}")
        else:
            reported = _finite_number(value, f"{source}/{name}")
            if not math.isclose(reported, float(actual[name]), rel_tol=tolerance, abs_tol=tolerance):
                _fail(f"metric mismatch for {source}/{name}: reported={reported!r}, recomputed={actual[name]!r}")


def validate_ranking_record(record: Mapping[str, Any], task: Mapping[str, Any], source: str = "ranking") -> dict[str, Any]:
    if not isinstance(record, Mapping) or not isinstance(task, Mapping):
        _fail(f"ranking/task must be objects: {source}")
    if record.get("task_id") != task.get("task_id"):
        _fail(f"task_id mismatch for {source}: {record.get('task_id')!r} != {task.get('task_id')!r}")
    depth = record.get("depth", task.get("depth"))
    if type(depth) is not int or depth not in EXPECTED_CANDIDATES:
        _fail(f"invalid depth for {source}: {depth!r}")
    if task.get("depth") != depth:
        _fail(f"task/ranking depth mismatch for {source}")
    for field in ("task_fingerprint", "split", "p", "start", "target"):
        if field in record and field in task and record[field] != task[field]:
            _fail(f"task/ranking {field} mismatch for {source}")
    entries = record.get("ranking")
    if not isinstance(entries, list):
        _fail(f"ranking is not an array for {source}")
    expected_count = EXPECTED_CANDIDATES[depth]
    if len(entries) != expected_count:
        _fail(f"full ranking size mismatch for {source}: {len(entries)} != {expected_count}")
    expected_programs = set(enumerate_programs(depth))
    programs: list[tuple[str, ...]] = []
    scores: list[float] = []
    target = tuple(task.get("target", ()))
    start = tuple(task.get("start", ()))
    field = task.get("p")
    if not start or not target or len(start) != len(target):
        _fail(f"invalid task state/target for {source}")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            _fail(f"ranking[{index}] is not an object for {source}")
        raw_program = entry.get("program")
        if not isinstance(raw_program, list) or len(raw_program) != depth or any(op not in OPS for op in raw_program):
            _fail(f"invalid program at {source}[{index}]: {raw_program!r}")
        program = tuple(raw_program)
        programs.append(program)
        scores.append(_finite_number(entry.get("score"), f"{source}[{index}].score"))
        exact = apply_program(start, program, field) == target
        if type(entry.get("correct")) is not bool or entry.get("correct") != exact:
            _fail(f"incorrect exact label at {source}[{index}] for program {program}")
        if "rank" in entry and (type(entry["rank"]) is not int or entry["rank"] != index + 1):
            _fail(f"incorrect explicit rank at {source}[{index}]: {entry.get('rank')!r}")
    if len(set(programs)) != expected_count or set(programs) != expected_programs:
        _fail(f"ranking is not the exact unique 5^{depth} program enumeration for {source}")
    expected_order = sorted(range(len(entries)), key=lambda i: (-scores[i], programs[i]))
    if expected_order != list(range(len(entries))):
        _fail(f"ranking order is not score-descending with lexical tie-break for {source}")
    metrics = recompute_metrics(entries)
    embedded = _metric_fields(record)
    if embedded:
        expected_names = set(METRIC_NAMES)
        if set(embedded) != expected_names:
            _fail(f"partial embedded metrics are forbidden for {source}: {sorted(embedded)}")
        assert_metrics_match(metrics, embedded, source)
    return {"task_id": record["task_id"], "depth": depth, **metrics}


def load_task_index(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(map(Path, paths)):
        for line_number, row in iter_jsonl(path):
            if not isinstance(row, dict):
                _fail(f"task record is not an object at {path}:{line_number}")
            schema = str(row.get("schema", ""))
            if "trajectory" in schema or not {"task_id", "p", "depth", "start", "target"}.issubset(row):
                continue
            task_id = row["task_id"]
            if not isinstance(task_id, str) or not task_id:
                _fail(f"invalid task_id at {path}:{line_number}")
            if task_id in tasks:
                _fail(f"duplicate task_id across task data: {task_id}")
            tasks[task_id] = row
    if not tasks:
        _fail("no task records found")
    return tasks


def discover_task_files(root: Path) -> list[Path]:
    data = Path(root) / "data"
    if not data.is_dir():
        return []
    return sorted([*data.rglob("*.jsonl"), *data.rglob("*.jsonl.gz")])


def _metric_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    binding = row.get("binding")
    seed = row.get("seed", binding.get("seed") if isinstance(binding, Mapping) else None)
    return seed, row.get("branch"), row.get("split"), row.get("task_id")


def load_reported_metrics(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line_number, row in iter_jsonl(path):
        if not isinstance(row, dict):
            _fail(f"metric record is not an object at {path}:{line_number}")
        key = _metric_key(row)
        if any(value is None for value in key) or key in result:
            _fail(f"missing or duplicate metric identity at {path}:{line_number}: {key}")
        result[key] = row
    return result


def metrics_sibling(ranking_path: Path) -> Path:
    ranking_path = Path(ranking_path)
    if not ranking_path.name.endswith(".jsonl.gz"):
        _fail(f"full ranking filename must end in .jsonl.gz: {ranking_path}")
    parent_parts = list(ranking_path.parent.parts)
    indices = [index for index, part in enumerate(parent_parts) if part == "rankings"]
    if indices:
        index = indices[-1]
        metric_parent = Path(*parent_parts[:index], "metrics", *parent_parts[index + 1:])
        name = ranking_path.name[:-3]
        if name.endswith(".rankings.jsonl"):
            name = name[:-len(".rankings.jsonl")] + ".metrics.jsonl"
        return metric_parent / name
    suffix = ".rankings.jsonl.gz"
    if ranking_path.name.endswith(suffix):
        return ranking_path.with_name(ranking_path.name[: -len(suffix)] + ".metrics.jsonl")
    _fail(f"ranking path is not under a rankings directory: {ranking_path}")


def receipt_sibling(metrics_path: Path) -> Path:
    metrics_path = Path(metrics_path)
    suffix = ".jsonl"
    if not metrics_path.name.endswith(suffix):
        _fail(f"metrics filename must end in {suffix}: {metrics_path}")
    return metrics_path.with_name(metrics_path.name[:-len(suffix)] + ".receipt.json")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReleaseValidationError(f"value cannot be canonically encoded: {exc}") from exc


def validate_ranking_file(
    path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
    metrics_path: Path | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.name.endswith(".jsonl.gz") or ("rankings" not in path.parts and not path.name.endswith(".rankings.jsonl.gz")):
        _fail(f"full ranking must be gzip JSONL: {path}")
    reported = load_reported_metrics(metrics_path) if metrics_path is not None and Path(metrics_path).exists() else None
    seen: set[str] = set()
    recomputed: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    branches: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    seeds: Counter[str] = Counter()
    depths: Counter[int] = Counter()
    task_ids_in_order: list[str] = []
    shard_binding: Any = None
    for line_number, row in iter_jsonl(path):
        if not isinstance(row, dict):
            _fail(f"ranking record is not an object at {path}:{line_number}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id in seen:
            _fail(f"missing or duplicate ranking task_id at {path}:{line_number}: {task_id!r}")
        if task_id not in tasks:
            _fail(f"ranking references unknown task_id at {path}:{line_number}: {task_id}")
        seen.add(task_id)
        task_ids_in_order.append(task_id)
        metric = validate_ranking_record(row, tasks[task_id], f"{path}:{line_number}")
        identity = _metric_key(row)
        if any(value is None for value in identity):
            _fail(f"ranking identity lacks seed/branch/split/task_id at {path}:{line_number}")
        identities.add(identity)
        if reported is not None:
            if identity not in reported:
                _fail(f"reported metrics missing ranking identity {identity} in {metrics_path}")
            reported_row = reported[identity]
            reported_metrics = _metric_fields(reported_row)
            if set(reported_metrics) != set(METRIC_NAMES):
                _fail(f"reported metrics are incomplete for {identity}: {sorted(reported_metrics)}")
            for field_name in ("task_fingerprint", "branch", "split", "task_id", "p", "depth", "binding"):
                if field_name in row and field_name in reported_row and row[field_name] != reported_row[field_name]:
                    _fail(f"ranking/metric {field_name} mismatch for {identity}")
            assert_metrics_match(metric, reported_metrics, f"{metrics_path}:{identity}")
        binding = row.get("binding")
        if isinstance(binding, Mapping):
            if binding.get("seed") is not None and int(binding["seed"]) != int(identity[0]):
                _fail(f"ranking/binding seed mismatch at {path}:{line_number}")
            if binding.get("split") is not None and binding.get("split") != row.get("split"):
                _fail(f"ranking/binding split mismatch at {path}:{line_number}")
            canonical_branch = binding.get("branch")
            if canonical_branch is not None:
                if canonical_branch not in CONFIRM_BRANCHES:
                    _fail(f"unknown canonical branch in ranking binding at {path}:{line_number}: {canonical_branch!r}")
                expected_labels = {
                    canonical_branch,
                    f"seed{int(identity[0])}_{canonical_branch}_{row.get('split')}",
                }
                if row.get("branch") not in expected_labels:
                    _fail(f"ranking branch is inconsistent with binding at {path}:{line_number}")
        if shard_binding is None:
            shard_binding = binding
        elif _canonical_json(binding) != _canonical_json(shard_binding):
            _fail(f"mixed bindings in ranking shard {path}")
        recomputed.append(metric)
        branches[str(row["branch"])] += 1
        splits[str(row["split"])] += 1
        seeds[str(identity[0])] += 1
        depths[int(metric["depth"])] += 1
    if reported is not None and set(reported) != identities:
        _fail(f"reported metrics contain identities absent from rankings: {set(reported) - identities}")
    if len(branches) != 1 or len(splits) != 1 or len(seeds) != 1:
        _fail(f"ranking shard mixes branch/split/seed identities: {path}")
    if receipt_path is not None:
        receipt_path = Path(receipt_path)
        receipt = load_json(receipt_path)
        if not isinstance(receipt, dict) or receipt.get("status") != "DONE":
            _fail(f"evaluation shard receipt is absent or not DONE: {receipt_path}")
        if receipt.get("task_ids") != task_ids_in_order:
            _fail(f"evaluation receipt task order mismatch: {receipt_path}")
        if receipt.get("branch") != next(iter(branches)):
            _fail(f"evaluation receipt branch mismatch: {receipt_path}")
        if _canonical_json(receipt.get("binding")) != _canonical_json(shard_binding):
            _fail(f"evaluation receipt binding mismatch: {receipt_path}")
        if str(receipt.get("ranking_sha256", "")).upper() != sha256_file(path):
            _fail(f"evaluation receipt ranking SHA-256 mismatch: {receipt_path}")
        if metrics_path is None or str(receipt.get("metrics_sha256", "")).upper() != sha256_file(Path(metrics_path)):
            _fail(f"evaluation receipt metrics SHA-256 mismatch: {receipt_path}")
        if receipt.get("ranking_rows") != len(recomputed) or receipt.get("metrics_rows") != len(recomputed):
            _fail(f"evaluation receipt row-count mismatch: {receipt_path}")
    means = {
        name: math.fsum(float(row[name]) for row in recomputed) / len(recomputed)
        for name in [*(f"hit@{cutoff}" for cutoff in HIT_CUTOFFS), "correct_mass", "mrr", "log_gap"]
    }
    means["best_rank_mean"] = math.fsum(row["best_rank"] for row in recomputed) / len(recomputed)
    binding_map = shard_binding if isinstance(shard_binding, Mapping) else {}
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows": len(recomputed),
        "branches": dict(branches),
        "splits": dict(splits),
        "seeds": dict(seeds),
        "depths": {str(key): value for key, value in sorted(depths.items())},
        "means": means,
        "seed": next(iter(seeds)),
        "split": next(iter(splits)),
        "canonical_branch": binding_map.get("branch", next(iter(branches))),
        "attempt": binding_map.get("attempt"),
        "binding": dict(binding_map),
        "_task_ids": task_ids_in_order,
    }


def discover_ranking_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.rglob("*.jsonl.gz")
                  if "rankings" in path.relative_to(root).parts and "archives" not in path.relative_to(root).parts)


def _validated_frozen_splits(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    frozen_path = root / "CONFIG_FROZEN.json"
    if not frozen_path.is_file():
        _fail("confirmatory release lacks CONFIG_FROZEN.json")
    frozen = load_json(frozen_path)
    if not isinstance(frozen, dict) or frozen.get("status") != "FROZEN":
        _fail("confirmatory configuration is not FROZEN")
    if tuple(frozen.get("confirm_seeds", ())) != CONFIRM_SEEDS:
        _fail(f"confirmatory seeds are not exactly {list(CONFIRM_SEEDS)}")
    test_files = frozen.get("test_files")
    if not isinstance(test_files, dict) or not test_files:
        _fail("CONFIG_FROZEN.json has no test_files mapping")
    if "confirm_atomic.jsonl" not in test_files:
        _fail("CONFIG_FROZEN.json lacks confirm_atomic.jsonl")
    atomic_entry = test_files["confirm_atomic.jsonl"]
    atomic_path = root / "data/confirm_atomic.jsonl"
    if (not isinstance(atomic_entry, Mapping) or not atomic_path.is_file() or atomic_path.is_symlink() or
            type(atomic_entry.get("bytes")) is not int or atomic_entry["bytes"] != atomic_path.stat().st_size or
            not isinstance(atomic_entry.get("sha256"), str) or atomic_entry["sha256"].upper() != sha256_file(atomic_path)):
        _fail("frozen confirm_atomic hash/size mismatch")
    atomic_rows = load_jsonl(atomic_path)
    if type(atomic_entry.get("rows")) is not int or atomic_entry["rows"] != len(atomic_rows):
        _fail("frozen confirm_atomic row-count mismatch")
    selection_path = root / "CONFIG_SELECTION_FROZEN.json"
    protocol_path = root / "configs/protocol.json"
    if (not selection_path.is_file() or
            str(frozen.get("selection_sha256", "")).upper() != sha256_file(selection_path)):
        _fail("CONFIG_FROZEN selection SHA-256 mismatch")
    if (not protocol_path.is_file() or
            str(frozen.get("protocol_sha256", "")).upper() != sha256_file(protocol_path)):
        _fail("CONFIG_FROZEN protocol SHA-256 mismatch")
    frozen_code = frozen.get("code")
    current_code = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted((root / "code").glob("*.py"))
    }
    if frozen_code != current_code:
        _fail("CONFIG_FROZEN code manifest mismatch")
    final_manifest_path = root / "manifests/final_data_manifest.json"
    final_manifest = load_json(final_manifest_path) if final_manifest_path.is_file() else None
    if (not isinstance(final_manifest, Mapping) or final_manifest.get("status") != "FROZEN" or
            final_manifest.get("files") != test_files):
        _fail("final data manifest differs from CONFIG_FROZEN test_files")
    split_files = {name: entry for name, entry in test_files.items() if name != "confirm_atomic.jsonl"}
    if not split_files or any(not name.endswith(".jsonl") for name in split_files):
        _fail("CONFIG_FROZEN.json contains invalid frozen ranking split names")
    expected_filenames = {f"{split}.jsonl" for split in CONFIRM_FROZEN_SPLITS}
    if set(split_files) != expected_filenames:
        _fail(f"frozen ranking split set mismatch: missing={sorted(expected_filenames-set(split_files))}, extra={sorted(set(split_files)-expected_filenames)}")

    tasks: dict[str, set[str]] = {}
    entries: dict[str, dict[str, Any]] = {}
    for filename, entry in sorted(split_files.items()):
        if not isinstance(entry, Mapping):
            _fail(f"invalid frozen test manifest entry: {filename}")
        path = root / "data" / filename
        if not path.is_file() or path.is_symlink():
            _fail(f"missing frozen test file: {path}")
        if (type(entry.get("bytes")) is not int or entry["bytes"] != path.stat().st_size or
                not isinstance(entry.get("sha256"), str) or entry["sha256"].upper() != sha256_file(path)):
            _fail(f"frozen test hash/size mismatch: {filename}")
        rows = load_jsonl(path)
        if type(entry.get("rows")) is not int or entry["rows"] != len(rows):
            _fail(f"frozen test row-count mismatch: {filename}")
        split = filename[:-len(".jsonl")]
        ids: set[str] = set()
        for index, row in enumerate(rows, 1):
            if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str):
                _fail(f"invalid task identity in frozen split {filename}:{index}")
            if row.get("split") != split:
                _fail(f"frozen task split mismatch in {filename}:{index}")
            if row["task_id"] in ids:
                _fail(f"duplicate task_id in frozen split {filename}: {row['task_id']}")
            ids.add(row["task_id"])
        tasks[split] = ids
        entries[split] = dict(entry)
    return {
        "frozen": frozen,
        "config_sha256": sha256_file(frozen_path),
        "tasks": tasks,
        "entries": entries,
    }


def validate_confirm_ranking_matrix(root: Path, ranking_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frozen = _validated_frozen_splits(root)
    expected_tasks: Mapping[str, set[str]] = frozen["tasks"]
    expected_keys = {
        (seed, branch, split)
        for split in expected_tasks
        for seed in CONFIRM_SEEDS
        for branch in CONFIRM_BRANCHES
    }
    coverage: dict[tuple[int, str, str], set[str]] = {}
    shard_counts: Counter[tuple[int, str, str]] = Counter()
    confirm_reports = 0
    for report in ranking_reports:
        attempt = report.get("attempt")
        split = str(report.get("split"))
        if attempt != "confirm_frozen":
            if split in expected_tasks:
                _fail(f"frozen split ranking is not bound to confirm_frozen: {report.get('path')}")
            continue
        confirm_reports += 1
        binding = report.get("binding")
        if not isinstance(binding, Mapping):
            _fail(f"confirm ranking lacks binding: {report.get('path')}")
        try:
            seed = int(binding.get("seed"))
        except (TypeError, ValueError):
            _fail(f"confirm ranking has invalid seed: {report.get('path')}")
        branch = binding.get("branch")
        if binding.get("split") != split or (seed, branch, split) not in expected_keys:
            _fail(f"unexpected confirm ranking identity {(seed, branch, split)}: {report.get('path')}")
        if str(binding.get("config_frozen_sha256", "")).upper() != frozen["config_sha256"]:
            _fail(f"confirm ranking CONFIG_FROZEN binding mismatch: {report.get('path')}")
        if str(binding.get("data_sha256", "")).upper() != frozen["entries"][split]["sha256"].upper():
            _fail(f"confirm ranking data binding mismatch: {report.get('path')}")
        key = (seed, str(branch), split)
        cell = coverage.setdefault(key, set())
        task_ids = report.get("_task_ids")
        if not isinstance(task_ids, list) or any(not isinstance(task_id, str) for task_id in task_ids):
            _fail(f"confirm ranking report lacks task coverage: {report.get('path')}")
        overlap = cell.intersection(task_ids)
        if overlap:
            _fail(f"duplicate task coverage in confirm cell {key}: {sorted(overlap)[:3]}")
        cell.update(task_ids)
        shard_counts[key] += 1
    if not confirm_reports:
        _fail("confirmatory release contains no confirm_frozen rankings")
    actual_keys = set(coverage)
    if actual_keys != expected_keys:
        _fail(f"incomplete confirm ranking matrix: missing={sorted(expected_keys-actual_keys)}, extra={sorted(actual_keys-expected_keys)}")
    for key in sorted(expected_keys):
        expected = expected_tasks[key[2]]
        actual = coverage[key]
        if actual != expected:
            _fail(f"confirm task coverage mismatch for {key}: missing={len(expected-actual)}, extra={len(actual-expected)}")
    rows = sum(len(task_ids) for task_ids in coverage.values())
    expected_rows = sum(len(task_ids) for task_ids in expected_tasks.values()) * len(CONFIRM_SEEDS) * len(CONFIRM_BRANCHES)
    if rows != expected_rows:
        _fail(f"confirm ranking matrix row-count mismatch: {rows} != {expected_rows}")
    return {
        "schema": "stage4.composition.v5.confirm-ranking-matrix.v1",
        "status": "PASS",
        "seeds": list(CONFIRM_SEEDS),
        "branches": list(CONFIRM_BRANCHES),
        "splits": {split: len(task_ids) for split, task_ids in sorted(expected_tasks.items())},
        "cells": len(expected_keys),
        "ranking_rows": rows,
        "shards": sum(shard_counts.values()),
        "config_frozen_sha256": frozen["config_sha256"],
    }


def _normalise_run_record(record: Mapping[str, Any], source: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        _fail(f"run record is not an object: {source}")
    run_id = record.get("run_id", record.get("id"))
    status = record.get("status", record.get("training_status"))
    if not isinstance(run_id, str) or not run_id or not isinstance(status, str) or not status:
        _fail(f"run record lacks non-empty run_id/status: {source}")
    out = dict(record)
    if isinstance(out.get("schema"), str) and out["schema"] != "stage4.composition.v5.run.v1":
        out["source_schema"] = out["schema"]
    out["schema"] = "stage4.composition.v5.run.v1"
    out["run_id"] = run_id
    out["status"] = status
    out.setdefault("source", source)
    out.setdefault("kind", "run")
    out.pop("id", None)
    return out


_RUN_INDEX_FIELDS = {
    "schema", "source_schema", "run_id", "status", "source", "kind", "source_sha256", "source_bytes",
    "seed", "branch", "phase", "attempt_id", "pass", "checks", "budget_checks", "resolved_config",
    "initial_attempt", "selected_attempt", "attempts", "dev_cycle", "final_created",
    "composition_confirm_unlocked", "composition_started", "final_composition_created",
    "scientific_result", "scientific_outcome", "positive_A", "positive_B", "model",
    "statistics_sha256", "config_frozen_sha256", "pilot_decision_sha256", "atomic_decision_sha256",
    "optimizer_steps", "optimizer_steps_expected", "loss_bearing_target_tokens_seen",
    "loss_bearing_target_tokens_expected", "effective_batch", "epochs", "wall_seconds", "peak_gpu_bytes",
    "input_sha256", "resolved_config_sha256", "adapter_tree_sha256", "binding", "tasks",
    "error_type", "error", "scientific_cycle_consumed", "generation_sha256", "gate_sha256",
    "started_at_unix", "finished_at_unix", "completed_at_unix",
}


def _compact_run_index_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key in _RUN_INDEX_FIELDS}


def discover_run_records(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    records: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    def add(path: Path, *, kind: str, run_id: str | None = None) -> None:
        source = path.relative_to(root).as_posix()
        if source in seen_sources:
            return
        value = load_json(path)
        if not isinstance(value, Mapping):
            _fail(f"run/status record is not an object: {path}")
        derived = dict(value)
        if run_id is not None:
            derived["run_id"] = run_id
        derived["kind"] = kind
        derived["source_sha256"] = sha256_file(path)
        derived["source_bytes"] = path.stat().st_size
        records.append(_normalise_run_record(derived, source))
        seen_sources.add(source)

    candidates = sorted({*root.glob("runs/**/run_record.json"), *root.glob("runs/**/run.json")})
    for path in candidates:
        kind = "release_record" if path.relative_to(root).as_posix() == "runs/terminal/run.json" else "run"
        add(path, kind=kind)
    for path in sorted(root.glob("runs/pilot/*/attempt.json")):
        value = load_json(path)
        attempt_id = value.get("attempt_id") if isinstance(value, Mapping) else None
        if not isinstance(attempt_id, str) or not attempt_id:
            _fail(f"pilot attempt lacks attempt_id: {path}")
        add(path, kind="pilot_attempt", run_id=f"pilot::{attempt_id}")
    pilot_decision = root / "runs/PILOT_DECISION.json"
    if pilot_decision.is_file():
        add(pilot_decision, kind="pilot_decision", run_id="pilot-decision")
    for name, fallback in (("STAGE4_DONE.json", "stage4-terminal-done"), ("STAGE4_FAILED.json", "stage4-terminal-failed")):
        path = root / "runs" / name
        if path.is_file():
            value = load_json(path)
            run_id = value.get("run_id", fallback) if isinstance(value, Mapping) else fallback
            add(path, kind="terminal_result", run_id=run_id)
    for path in sorted(root.glob("runs/failed/*.json")):
        value = load_json(path)
        run_id = value.get("run_id", f"failed::{path.stem}") if isinstance(value, Mapping) else f"failed::{path.stem}"
        add(path, kind="infrastructure_failure", run_id=run_id)
    prepare = root / "runs/PREPARE_DONE.json"
    if prepare.is_file():
        add(prepare, kind="phase", run_id="prepare")
    for path in sorted(root.glob("adapters/**/training_receipt.json")):
        receipt = load_json(path)
        if not isinstance(receipt, dict):
            _fail(f"training receipt is not an object: {path}")
        adapter_identity = path.parent.relative_to(root / "adapters").as_posix().replace("/", "::")
        derived = {**receipt, "run_id": receipt.get("run_id", f"adapter::{adapter_identity}"), "status": receipt.get("status"),
                   "source": path.relative_to(root).as_posix(), "kind": "training",
                   "source_sha256": sha256_file(path), "source_bytes": path.stat().st_size}
        records.append(_normalise_run_record(derived, path.relative_to(root).as_posix()))
        seen_sources.add(path.relative_to(root).as_posix())
    status_paths = sorted({
        *root.glob("runs/**/status/*.json"),
        *root.glob("runs/training/*.status.json"),
        *root.glob("adapters/**/training_status.json"),
    })
    final_access = root / "manifests/final_evaluation_access.json"
    if final_access.is_file():
        status_paths.append(final_access)
    for path in status_paths:
        add(path, kind="status", run_id=f"status::{path.relative_to(root).as_posix()}")
    return records


def write_run_index(records: Sequence[Mapping[str, Any]], output: Path) -> dict[str, Any]:
    normalised = [
        _compact_run_index_record(_normalise_run_record(record, f"record[{index}]"))
        for index, record in enumerate(records)
    ]
    if not normalised:
        _fail("cannot build an empty run index")
    ids = [record["run_id"] for record in normalised]
    if len(ids) != len(set(ids)):
        duplicates = sorted(run_id for run_id, count in Counter(ids).items() if count > 1)
        _fail(f"duplicate run ids: {duplicates}")
    normalised.sort(key=lambda row: row["run_id"])
    counts = Counter(record["status"] for record in normalised)
    kind_counts = Counter(record.get("kind", "run") for record in normalised)
    pilot_attempts = [record for record in normalised if record.get("kind") == "pilot_attempt"]
    pilot_decisions = [record for record in normalised if record.get("kind") == "pilot_decision"]
    terminal_results = [record for record in normalised if record.get("kind") == "terminal_result"]
    statuses = [record for record in normalised if record.get("kind") == "status"]
    if len(pilot_decisions) > 1:
        _fail(f"multiple pilot decisions in run index: {[row['source'] for row in pilot_decisions]}")
    if len(terminal_results) > 1:
        _fail(f"multiple authoritative terminal results in run index: {[row['source'] for row in terminal_results]}")
    index = {
        "schema": "stage4.composition.v5.run-index.v1",
        "runs": normalised,
        "run_count": len(normalised),
        "status_counts": dict(sorted(counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "pilot_attempts": pilot_attempts,
        "pilot_attempt_count": len(pilot_attempts),
        "pilot_decision": pilot_decisions[0] if pilot_decisions else None,
        "statuses": statuses,
        "status_record_count": len(statuses),
        "terminal": terminal_results[0] if terminal_results else None,
    }
    local_schema = output.parents[1] / "schemas/run_index.schema.json"
    validate_value(index, load_json(local_schema if local_schema.exists() else SCHEMAS / "run_index.schema.json"), str(output))
    write_json(output, index)
    return index


def discover_atomic_run_records(atomic_root: Path) -> list[dict[str, Any]]:
    atomic_root = Path(atomic_root).resolve(); records = []

    def add(value: Mapping[str, Any], path: Path, run_id: str, kind: str, *, status: str | None = None) -> None:
        source = f"{atomic_root.name}/{path.relative_to(atomic_root).as_posix()}"
        record = {**dict(value), "run_id": run_id, "status": status or value.get("status"), "source": source,
                  "kind": kind, "source_sha256": sha256_file(path), "source_bytes": path.stat().st_size}
        records.append(_normalise_run_record(record, source))

    for path in sorted((atomic_root / "adapters").glob("*/training_receipt.json")):
        receipt = load_json(path); phase = receipt.get("phase", {})
        add({**phase, "status": receipt.get("status")}, path, f"atomic::train::{phase.get('phase', path.parent.name)}", "atomic_training")
    for run_id in ("curriculum_calibration", "corrective_calibration", "atomic_v5_final"):
        path = atomic_root / "runs" / f"{run_id}_gate.json"
        if path.is_file():
            gate = load_json(path)
            add({**gate, "phase": "evaluation", "status": "DONE"}, path, f"atomic::eval::{run_id}", "atomic_evaluation")
    for path in sorted((atomic_root / "runs/failed").glob("*.json")):
        failure = load_json(path)
        add(failure, path, f"atomic::failed::{path.stem}", "atomic_failed")
    for path in sorted((atomic_root / "runs/superseded").glob("**/*_gate.json")):
        gate = load_json(path)
        add({**gate, "status": "SUPERSEDED", "phase": "evaluation"}, path,
            f"atomic::superseded::{path.parent.name}", "atomic_superseded")
    audit_path = atomic_root / "manifests/plan_raw_audit.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
        for row in audit.get("runs", []):
            add({**row, "status": "DONE", "phase": "plan_raw_audit"}, audit_path,
                f"atomic::audit::plan::{row.get('run_id')}", "atomic_audit")
    decision_path = atomic_root / "runs/ATOMIC_DECISION.json"
    if decision_path.is_file():
        decision = load_json(decision_path)
        add(decision, decision_path, "atomic::decision", "atomic_decision")
    return records


def build_run_index(root: Path, atomic_root: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    records = discover_run_records(root)
    if atomic_root is not None:
        records.extend(discover_atomic_run_records(atomic_root))
    existing = root / "manifests/RUN_INDEX.json"
    if not records and existing.exists():
        value = validate_json_file(existing, root / "schemas/run_index.schema.json")
        return value
    return write_run_index(records, existing)


def _safe_reference_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        _fail(f"invalid reference path: {value!r}")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        _fail(f"atomic reference must be normalized and repo-relative: {value!r}")
    if path.suffix.lower() == ".zip" or "archives" in {part.lower() for part in path.parts}:
        _fail(f"old archives may not be referenced: {value!r}")
    if "stage4_sh1_v5_0p6b" not in path.parts:
        _fail(f"reference is not bound to atomic v5: {value!r}")
    return path


def _reference_items(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if isinstance(files, list):
        return [dict(item) for item in files if isinstance(item, Mapping)]
    if isinstance(files, dict):
        result = []
        for path, metadata in files.items():
            if not isinstance(metadata, Mapping):
                _fail(f"invalid atomic reference metadata for {path!r}")
            result.append({"path": path, **dict(metadata)})
        return result
    _fail("atomic reference manifest files must be an array or object")


def _tree_sha256(path: Path) -> str:
    path = Path(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        _fail(f"cannot hash empty tree: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            _fail(f"atomic adapter tree contains a symlink: {item}")
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest().upper()


def validate_atomic_reference_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    path = Path(path)
    manifest = validate_json_file(path, path.parents[1] / "schemas/atomic_reference.schema.json")
    repo_root = Path(repo_root).resolve()
    checked = []
    checked_paths: set[str] = set()
    for item in _reference_items(manifest):
        rel = _safe_reference_path(item.get("path"))
        if rel.as_posix() in checked_paths:
            _fail(f"duplicate atomic reference path: {rel}")
        nominal = repo_root / Path(*rel.parts)
        target = nominal.resolve()
        try:
            target.relative_to(repo_root)
        except ValueError:
            _fail(f"atomic reference escapes repository: {rel}")
        if not target.is_file() or nominal.is_symlink() or target.is_symlink():
            _fail(f"missing atomic reference: {target}")
        expected_bytes = item.get("bytes")
        expected_sha = item.get("sha256")
        if type(expected_bytes) is not int or expected_bytes != target.stat().st_size:
            _fail(f"atomic reference size mismatch: {rel}")
        if not isinstance(expected_sha, str) or expected_sha.upper() != sha256_file(target):
            _fail(f"atomic reference SHA-256 mismatch: {rel}")
        checked.append({"path": rel.as_posix(), "bytes": expected_bytes, "sha256": expected_sha.upper()})
        checked_paths.add(rel.as_posix())
    if not checked:
        _fail("atomic reference manifest is empty")
    decisions = [item for item in checked if PurePosixPath(item["path"]).name == "ATOMIC_DECISION.json"]
    if len(decisions) != 1:
        _fail(f"atomic reference manifest must contain exactly one ATOMIC_DECISION.json, found {len(decisions)}")
    decision_path = repo_root / Path(*PurePosixPath(decisions[0]["path"]).parts)
    decision = load_json(decision_path)
    if not isinstance(decision, dict) or not isinstance(decision.get("status"), str):
        _fail("referenced atomic decision lacks a terminal status")

    adapter_metadata = manifest.get("selected_adapter")
    checked_adapter: dict[str, Any] | None = None
    if adapter_metadata is not None:
        if not isinstance(adapter_metadata, Mapping):
            _fail("selected_adapter metadata is not an object")
        adapter_rel = _safe_reference_path(adapter_metadata.get("path"))
        adapter_dir = (repo_root / Path(*adapter_rel.parts)).resolve()
        try:
            adapter_dir.relative_to(repo_root)
        except ValueError:
            _fail(f"selected atomic adapter escapes repository: {adapter_rel}")
        if not adapter_dir.is_dir() or adapter_dir.is_symlink():
            _fail(f"selected atomic adapter is missing or unsafe: {adapter_dir}")
        adapter_files = sorted(item for item in adapter_dir.rglob("*") if item.is_file())
        for item in adapter_dir.rglob("*"):
            if item.is_symlink():
                _fail(f"selected atomic adapter contains a symlink: {item}")
        adapter_paths = {item.relative_to(repo_root).as_posix() for item in adapter_files}
        referenced_adapter_paths = {item["path"] for item in checked if item["path"] in adapter_paths}
        if referenced_adapter_paths != adapter_paths:
            _fail(f"atomic adapter payload references are incomplete: missing={sorted(adapter_paths-referenced_adapter_paths)}")
        if (adapter_metadata.get("files_count") != len(adapter_files) or
                adapter_metadata.get("bytes") != sum(item.stat().st_size for item in adapter_files)):
            _fail("selected atomic adapter file-count/byte totals mismatch")
        if str(adapter_metadata.get("tree_sha256", "")).upper() != _tree_sha256(adapter_dir):
            _fail("selected atomic adapter tree SHA-256 mismatch")
        receipt_rel = _safe_reference_path(adapter_metadata.get("receipt_path"))
        if adapter_rel not in receipt_rel.parents:
            _fail("selected atomic adapter receipt is outside adapter tree")
        if receipt_rel.as_posix() not in checked_paths:
            _fail("selected atomic adapter receipt is absent from files references")
        receipt_path = repo_root / Path(*receipt_rel.parts)
        if str(adapter_metadata.get("receipt_sha256", "")).upper() != sha256_file(receipt_path):
            _fail("selected atomic adapter receipt SHA-256 mismatch")
        receipt = load_json(receipt_path)
        if not isinstance(receipt, dict) or receipt.get("status") != "DONE":
            _fail("selected atomic adapter receipt is not DONE")
        checked_adapter = {
            "path": adapter_rel.as_posix(),
            "tree_sha256": _tree_sha256(adapter_dir),
            "receipt_path": receipt_rel.as_posix(),
            "receipt_sha256": sha256_file(receipt_path),
            "files_count": len(adapter_files),
            "bytes": sum(item.stat().st_size for item in adapter_files),
        }
    if decision.get("status") == "PASS" and checked_adapter is None:
        _fail("atomic PASS reference manifest lacks selected_adapter payload")
    if decision.get("status") != "PASS" and checked_adapter is not None:
        _fail("failed atomic decision unexpectedly publishes a selected adapter")
    return {
        "path": path.as_posix(),
        "files": checked,
        "atomic_decision": {"path": decisions[0]["path"], "status": decision["status"], "sha256": decisions[0]["sha256"]},
        "selected_adapter": checked_adapter,
    }


def find_atomic_reference_manifest(root: Path) -> Path:
    manifests = Path(root) / "manifests"
    candidates = sorted({
        manifests / "atomic_v5_references.json",
        manifests / "atomic_reference.json",
        *manifests.glob("*atomic*reference*.json"),
    })
    matches = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        value = load_json(candidate)
        if isinstance(value, dict) and value.get("schema") == "stage4.composition.v5.atomic-references.v1":
            matches.append(candidate)
    if len(matches) != 1:
        _fail(f"expected exactly one atomic v5 reference manifest, found {len(matches)}: {matches}")
    return matches[0]


def _source_file(root: Path, path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    rel = PurePosixPath(path.relative_to(root).as_posix())
    if any(part in _CACHE_PARTS for part in rel.parts):
        return False
    if rel.parts[0] == "archives" or path.suffix.lower() == ".zip" or rel in _RECEIPT_PATHS:
        return False
    return True


def release_source_files(root: Path) -> list[Path]:
    root = Path(root).resolve()
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_symlink():
            _fail(f"release tree contains a symlink: {rel}")
        if path.is_file() and path.name.endswith((".partial", ".tmp")):
            _fail(f"release tree contains an incomplete temporary file: {rel}")
        if _source_file(root, path):
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def sibling_atomic_root(root: Path) -> Path | None:
    candidate = Path(root).resolve().parent / "stage4_sh1_v5_0p6b"
    if candidate.is_symlink():
        _fail(f"sibling atomic root may not be a symlink: {candidate}")
    if not candidate.exists():
        return None
    if not candidate.is_dir():
        _fail(f"sibling atomic root is not a directory: {candidate}")
    return candidate.resolve()


def _atomic_source_file(root: Path, path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    rel = PurePosixPath(path.relative_to(root).as_posix())
    lowered = tuple(part.lower() for part in rel.parts)
    if any(part in _CACHE_PARTS or part in {"cache", "caches"} for part in lowered):
        return False
    if any(part.endswith((".partial", ".tmp")) or part in {"partial", "tmp"} for part in lowered):
        return False
    if path.suffix.lower() == ".zip":
        return False
    return True


def atomic_payload_files(atomic_root: Path) -> list[Path]:
    atomic_root = Path(atomic_root).resolve()
    if atomic_root.name != "stage4_sh1_v5_0p6b" or not atomic_root.is_dir():
        _fail(f"invalid sibling atomic root: {atomic_root}")
    files = []
    for path in atomic_root.rglob("*"):
        rel = path.relative_to(atomic_root)
        if path.is_symlink():
            _fail(f"atomic release tree contains a symlink: {rel}")
        if _atomic_source_file(atomic_root, path):
            files.append(path)
    if not files:
        _fail(f"sibling atomic payload is empty: {atomic_root}")
    return sorted(files, key=lambda path: path.relative_to(atomic_root).as_posix())


def validate_atomic_payload_syntax(atomic_root: Path) -> dict[str, Any]:
    json_files = jsonl_files = jsonl_rows = 0
    for path in atomic_payload_files(atomic_root):
        if path.suffix.lower() == ".json":
            load_json(path); json_files += 1
        elif _is_jsonl(path):
            rows = sum(1 for _ in iter_jsonl(path)); jsonl_files += 1; jsonl_rows += rows
    return {"status": "PASS", "json_files": json_files, "jsonl_files": jsonl_files, "jsonl_rows": jsonl_rows}


def _payload_tree_sha256(files: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest().upper()


def build_atomic_payload_manifest(
    root: Path,
    atomic_root: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any] | None:
    root = Path(root).resolve()
    atomic_root = Path(atomic_root).resolve() if atomic_root is not None else sibling_atomic_root(root)
    output = Path(output or root / Path(*ATOMIC_PAYLOAD_MANIFEST.parts))
    if atomic_root is None:
        if output.exists():
            _fail(f"stale atomic payload manifest exists without sibling atomic root: {output}")
        return None
    payload = atomic_payload_files(atomic_root)
    files = {
        path.relative_to(atomic_root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in payload
    }
    manifest = {
        "schema": "stage4.composition.v5.atomic-payload-sha256.v1",
        "root": atomic_root.name,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "tree_sha256": _payload_tree_sha256(payload, atomic_root),
    }
    schema_path = root / "schemas/atomic_payload_sha256.schema.json"
    validate_value(manifest, load_json(schema_path), str(output))
    write_json(output, manifest)
    return manifest


def validate_atomic_payload_manifest(
    root: Path,
    atomic_root: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any] | None:
    root = Path(root).resolve()
    atomic_root = Path(atomic_root).resolve() if atomic_root is not None else sibling_atomic_root(root)
    manifest_path = Path(manifest_path or root / Path(*ATOMIC_PAYLOAD_MANIFEST.parts))
    if atomic_root is None:
        if manifest_path.exists():
            _fail(f"atomic payload manifest has no sibling root: {manifest_path}")
        return None
    manifest = validate_json_file(manifest_path, root / "schemas/atomic_payload_sha256.schema.json")
    payload = atomic_payload_files(atomic_root)
    expected = {path.relative_to(atomic_root).as_posix() for path in payload}
    recorded = set(manifest["files"])
    if expected != recorded:
        _fail(f"atomic payload manifest membership mismatch: missing={sorted(expected-recorded)}, extra={sorted(recorded-expected)}")
    for rel, item in manifest["files"].items():
        path = atomic_root / Path(*PurePosixPath(rel).parts)
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"].upper():
            _fail(f"atomic payload SHA-256 mismatch: {rel}")
    if (manifest["root"] != atomic_root.name or manifest["file_count"] != len(payload) or
            manifest["total_bytes"] != sum(path.stat().st_size for path in payload) or
            manifest["tree_sha256"].upper() != _payload_tree_sha256(payload, atomic_root)):
        _fail("atomic payload manifest aggregate binding mismatch")
    return manifest


def build_sha256_manifest(root: Path, output: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output or root / "manifests/SHA256SUMS.json")
    files: dict[str, dict[str, Any]] = {}
    for path in release_source_files(root):
        if path.resolve() == output.resolve():
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {"schema": "stage4.composition.v5.sha256.v1", "root": root.name, "files": files}
    validate_value(manifest, load_json(root / "schemas/sha256.schema.json"), str(output))
    write_json(output, manifest)
    return manifest


def validate_sha256_manifest(root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path or root / "manifests/SHA256SUMS.json").resolve()
    manifest = validate_json_file(manifest_path, root / "schemas/sha256.schema.json")
    expected_paths = {path.relative_to(root).as_posix() for path in release_source_files(root) if path.resolve() != manifest_path}
    recorded_paths = set(manifest["files"])
    if expected_paths != recorded_paths:
        _fail(f"SHA-256 manifest membership mismatch: missing={sorted(expected_paths-recorded_paths)}, extra={sorted(recorded_paths-expected_paths)}")
    for rel, item in manifest["files"].items():
        path = root / Path(*PurePosixPath(rel).parts)
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"].upper():
            _fail(f"SHA-256 manifest mismatch: {rel}")
    return manifest


def is_full_ranking(path: PurePosixPath) -> bool:
    return path.name.endswith(".jsonl.gz") and ("rankings" in path.parts or path.name.endswith(".rankings.jsonl.gz"))


def compact_includes(rel: PurePosixPath) -> bool:
    if rel.parts[0] == "adapters" or is_full_ranking(rel):
        return False
    if rel.parts[0] in {"reports", "configs", "manifests", "schemas", "code", "tests", "runs", "summary", "summaries"}:
        return True
    if "summary" in rel.name.lower():
        return True
    return len(rel.parts) == 1 and rel.name in {"PREREGISTRATION.md", "STATUS.md", "DELIVERY_INDEX.md", "BLOCKERS.md", ".gitignore"}


def atomic_compact_includes(rel: PurePosixPath) -> bool:
    if len(rel.parts) == 1:
        return rel.name in {"PREREGISTRATION.md", "STATUS.md", "DELIVERY_INDEX.md", "BLOCKERS.md", ".gitignore"}
    if rel.parts[0] in {"configs", "manifests", "reports", "runs", "code", "tests"}:
        return True
    if rel.parts[0] == "data":
        return rel.name == "calibration.jsonl"
    if rel.parts[0] == "adapters":
        return rel.name == "training_receipt.json"
    return False


def archive_member_lists(root: Path, atomic_root: Path | None = None) -> dict[str, list[str]]:
    root = Path(root).resolve()
    files = release_source_files(root)
    full = [path.relative_to(root).as_posix() for path in files]
    compact = [rel for rel in full if compact_includes(PurePosixPath(rel))]
    result = {"compact": compact, "full": full}
    atomic_root = Path(atomic_root).resolve() if atomic_root is not None else sibling_atomic_root(root)
    if atomic_root is not None:
        atomic_files = [path.relative_to(atomic_root).as_posix() for path in atomic_payload_files(atomic_root)]
        result["atomic_full"] = atomic_files
        result["atomic_compact"] = [rel for rel in atomic_files if atomic_compact_includes(PurePosixPath(rel))]
    return result


def _zip_member(root: Path, path: Path) -> str:
    return f"{root.name}/{path.relative_to(root).as_posix()}"


def _archive_membership_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{name}\n" for name in names).encode("utf-8")).hexdigest().upper()


def create_zip(
    root: Path,
    output: Path,
    members: Sequence[str],
    *,
    atomic_root: Path | None = None,
    atomic_members: Sequence[str] = (),
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(members))
    if len(unique) != len(members):
        _fail(f"duplicate archive member requested for {output}")
    atomic_unique = sorted(set(atomic_members))
    if len(atomic_unique) != len(atomic_members):
        _fail(f"duplicate atomic archive member requested for {output}")
    if atomic_unique and atomic_root is None:
        _fail("atomic archive members were provided without an atomic root")
    atomic_root = Path(atomic_root).resolve() if atomic_root is not None else None
    entries: list[tuple[str, Path, Any]] = []
    source_groups: list[tuple[Path, Sequence[str], Any]] = [(root, unique, _source_file)]
    if atomic_root is not None:
        source_groups.append((atomic_root, atomic_unique, _atomic_source_file))
    for source_root, rel_texts, predicate in source_groups:
        for rel_text in rel_texts:
            rel = PurePosixPath(rel_text)
            if rel.is_absolute() or ".." in rel.parts:
                _fail(f"unsafe archive member path: {rel_text}")
            source = source_root / Path(*rel.parts)
            if not predicate(source_root, source):
                _fail(f"archive member is absent or forbidden: {source}")
            entries.append((_zip_member(source_root, source), source, predicate))
    entries.sort(key=lambda item: item[0])
    expected = [name for name, _, _ in entries]
    if len(expected) != len(set(expected)):
        _fail(f"duplicate final archive member name requested for {output}")
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
            for member_name, source, _ in entries:
                info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with source.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if names != expected or len(names) != len(set(names)):
            _fail(f"archive membership/order mismatch: {output}")
        bad = archive.testzip()
        if bad is not None:
            _fail(f"archive CRC failure in {output}: {bad}")
    return {
        "path": output.relative_to(root).as_posix() if output.is_relative_to(root) else str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "crc_test": "PASS",
        "members": len(expected),
        "membership_sha256": _archive_membership_sha256(expected),
        "roots": dict(sorted(Counter(PurePosixPath(name).parts[0] for name in expected).items())),
    }


def create_release_archives(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifests = root / "manifests"
    archives = root / "archives"
    manifests.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)

    atomic_root = sibling_atomic_root(root)
    atomic_payload = build_atomic_payload_manifest(root, atomic_root)
    if atomic_payload is not None:
        validate_atomic_payload_manifest(root, atomic_root)

    # The content manifest and SHA manifest are themselves release members. List names only,
    # avoiding an impossible self-hash while making archive membership auditable.
    planned = archive_member_lists(root, atomic_root)
    for rel in ("manifests/archive_contents.json", "manifests/SHA256SUMS.json"):
        if rel not in planned["full"]:
            planned["full"].append(rel)
        if compact_includes(PurePosixPath(rel)) and rel not in planned["compact"]:
            planned["compact"].append(rel)
    planned = {name: sorted(values) for name, values in planned.items()}
    write_json(manifests / "archive_contents.json", {"schema": "stage4.composition.v5.archive-contents.v1", **planned})
    build_sha256_manifest(root)
    validate_sha256_manifest(root)
    if atomic_payload is not None:
        validate_atomic_payload_manifest(root, atomic_root)
    actual = archive_member_lists(root, atomic_root)
    if actual != planned:
        _fail(f"release tree changed while freezing archive membership: planned={planned}, actual={actual}")

    compact = create_zip(
        root,
        archives / "stage4_distill_v5_0p6b_audit_report.zip",
        planned["compact"],
        atomic_root=atomic_root,
        atomic_members=planned.get("atomic_compact", ()),
    )
    full = create_zip(
        root,
        archives / "stage4_distill_v5_0p6b_complete.zip",
        planned["full"],
        atomic_root=atomic_root,
        atomic_members=planned.get("atomic_full", ()),
    )
    validate_sha256_manifest(root)
    if atomic_payload is not None:
        atomic_payload = validate_atomic_payload_manifest(root, atomic_root)
    atomic_binding = None
    if atomic_payload is not None:
        atomic_manifest_path = root / Path(*ATOMIC_PAYLOAD_MANIFEST.parts)
        atomic_binding = {
            "path": ATOMIC_PAYLOAD_MANIFEST.as_posix(),
            "sha256": sha256_file(atomic_manifest_path),
            "root": atomic_payload["root"],
            "tree_sha256": atomic_payload["tree_sha256"],
            "file_count": atomic_payload["file_count"],
            "total_bytes": atomic_payload["total_bytes"],
        }
    receipt = {
        "schema": "stage4.composition.v5.archive-receipt.v1",
        "source_manifest": {"path": "manifests/SHA256SUMS.json", "sha256": sha256_file(manifests / "SHA256SUMS.json")},
        "atomic_payload_manifest": atomic_binding,
        "archives": {"compact": compact, "full": full},
    }
    validate_value(receipt, load_json(root / "schemas/archive_receipt.schema.json"), "archive receipt")
    write_json(manifests / "archive_receipt.json", receipt)
    write_json(archives / "RELEASE.json", receipt)
    return receipt


def _optional_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = load_json(path)
    if not isinstance(value, dict):
        _fail(f"terminal evidence is not an object: {path}")
    return value


def _validate_pilot_attempt_inventory(root: Path, decision: Mapping[str, Any]) -> dict[str, Any]:
    attempt_ids = decision.get("attempts")
    if not isinstance(attempt_ids, list) or not attempt_ids or any(not isinstance(item, str) or not item for item in attempt_ids):
        _fail("pilot decision has no complete attempts inventory")
    if len(attempt_ids) != len(set(attempt_ids)):
        _fail("pilot decision contains duplicate attempt ids")
    selected_id = decision.get("selected_attempt")
    if selected_id not in attempt_ids:
        _fail("pilot selected_attempt is absent from attempts inventory")
    attempts: dict[str, dict[str, Any]] = {}
    for attempt_id in attempt_ids:
        path = root / "runs" / "pilot" / attempt_id / "attempt.json"
        attempt = _optional_object(path)
        if attempt is None or attempt.get("status") != "DONE" or attempt.get("attempt_id") != attempt_id:
            _fail(f"pilot attempt is missing, mismatched, or not DONE: {attempt_id}")
        attempts[attempt_id] = attempt
    embedded = decision.get("selected")
    if not isinstance(embedded, Mapping) or _canonical_json(embedded) != _canonical_json(attempts[selected_id]):
        _fail("pilot decision selected payload differs from selected attempt receipt")
    return {"attempt_ids": sorted(attempt_ids), "selected_attempt": selected_id, "selected_pass": attempts[selected_id].get("pass")}


def validate_terminal_readiness(
    root: Path,
    atomic: Mapping[str, Any],
    confirm_matrix: Mapping[str, Any] | None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    done_path = root / "runs/STAGE4_DONE.json"
    failed_path = root / "runs/STAGE4_FAILED.json"
    done = _optional_object(done_path)
    failed = _optional_object(failed_path)
    if done is not None and failed is not None:
        _fail("both STAGE4_DONE.json and STAGE4_FAILED.json exist")
    pilot_path = root / "runs/PILOT_DECISION.json"
    pilot = _optional_object(pilot_path)
    atomic_status = atomic.get("atomic_decision", {}).get("status")

    if atomic_status != "PASS":
        if atomic_status not in {"FAILED_CALIBRATION_GATE", "FAILED_FINAL_GATE"}:
            _fail(f"unregistered atomic-negative decision status: {atomic_status!r}")
        if pilot is not None:
            _fail("atomic-negative release unexpectedly contains a pilot decision")
        if (done is None or done.get("status") != "DONE" or done.get("run_id") != "stage4-atomic-negative" or
                done.get("scientific_result") != "FAILED_ATOMIC_GATE" or done.get("composition_started") is not False or
                done.get("final_composition_created") is not False or
                str(done.get("atomic_decision_sha256", "")).upper() != str(atomic.get("atomic_decision", {}).get("sha256", "")).upper()):
            _fail("atomic-negative release lacks authoritative FAILED_ATOMIC_GATE terminal result")
        if failed is not None or (root / "CONFIG_FROZEN.json").exists() or confirm_matrix is not None:
            _fail("atomic-negative release contains composition/final evidence")
        allowed_run_files = {"runs/STAGE4_DONE.json", "runs/terminal/run.json"}
        for failure_path in sorted((root / "runs/failed").glob("*.json")):
            failure = load_json(failure_path)
            if (failure.get("schema") != "stage4.composition.v5.run.v1" or failure.get("status") != "FAILED" or
                    failure.get("phase") != "release_validation" or failure.get("composition_started") is not False or
                    failure.get("final_composition_created") is not False or failure.get("scientific_cycle_consumed") is not False):
                _fail(f"atomic-negative release contains an invalid infrastructure failure receipt: {failure_path}")
            allowed_run_files.add(failure_path.relative_to(root).as_posix())
        unexpected_runs = sorted(path.relative_to(root).as_posix() for path in (root / "runs").rglob("*")
                                 if path.is_file() and path.relative_to(root).as_posix() not in allowed_run_files)
        composition_payload = []
        for folder in ("data", "adapters", "rankings"):
            path = root / folder
            if path.exists(): composition_payload.extend(item.relative_to(root).as_posix() for item in path.rglob("*") if item.is_file())
        forbidden_files = [root / "CONFIG_SELECTION_FROZEN.json", root / "manifests/final_generation_receipt.json",
                           root / "manifests/final_evaluation_access.json", root / "reports/confirmatory_statistics.json"]
        if unexpected_runs or composition_payload or any(path.exists() for path in forbidden_files):
            _fail(f"atomic-negative release contains composition runtime payload: runs={unexpected_runs}, payload={sorted(composition_payload)}")
        return {
            "schema": "stage4.composition.v5.terminal-readiness.v1",
            "status": "PASS",
            "mode": "ATOMIC_NEGATIVE",
            "terminal_path": done_path.relative_to(root).as_posix(),
            "scientific_result": done["scientific_result"],
        }

    if pilot is None:
        _fail("atomic PASS release lacks PILOT_DECISION.json")
    pilot_inventory = _validate_pilot_attempt_inventory(root, pilot)
    pilot_status = pilot.get("status")
    if pilot_status == "FAILED_PILOT_GATE":
        if pilot_inventory["selected_pass"] is not False:
            _fail("FAILED_PILOT_GATE decision selected an attempt marked pass")
        if failed is None or failed.get("status") != "FAILED" or failed.get("phase") != "pilot_gate":
            _fail("failed pilot release lacks authoritative pilot-gate terminal result")
        if done is not None or confirm_matrix is not None or (root / "CONFIG_FROZEN.json").exists():
            _fail("failed pilot release contains confirmatory/frozen evidence")
        if pilot.get("final_created") is not False or pilot.get("composition_confirm_unlocked") is not False:
            _fail("failed pilot decision incorrectly unlocks final confirmation")
        return {
            "schema": "stage4.composition.v5.terminal-readiness.v1",
            "status": "PASS",
            "mode": "PILOT_NEGATIVE",
            "terminal_path": failed_path.relative_to(root).as_posix(),
            "scientific_result": failed.get("scientific_result"),
            "pilot": pilot_inventory,
        }
    if pilot_status != "PASS":
        _fail(f"pilot decision is not terminal: {pilot_status!r}")
    if pilot_inventory["selected_pass"] is not True:
        _fail("pilot PASS decision selected an attempt not marked pass")
    if pilot.get("final_created") is not True or pilot.get("composition_confirm_unlocked") is not True:
        _fail("pilot PASS did not freeze/unlock confirmatory evaluation")
    if failed is not None:
        _fail("pilot PASS release contains STAGE4_FAILED.json")
    if done is None or done.get("status") != "DONE" or done.get("run_id") != "stage4-confirmatory-series":
        _fail("pilot PASS may not be released before the confirmatory series is DONE")
    if confirm_matrix is None or confirm_matrix.get("status") != "PASS":
        _fail("pilot PASS may not be released without a complete confirm ranking matrix")
    frozen_path = root / "CONFIG_FROZEN.json"
    if str(pilot.get("config_frozen_sha256", "")).upper() != sha256_file(frozen_path):
        _fail("pilot decision CONFIG_FROZEN SHA-256 mismatch")
    frozen = load_json(frozen_path)
    generation_path = root / "manifests/final_generation_receipt.json"
    generation = _optional_object(generation_path)
    if (generation is None or generation.get("status") != "DONE" or generation.get("evaluation_count") != 1 or
            str(generation.get("config_frozen_sha256", "")).upper() != sha256_file(frozen_path) or
            generation.get("test_files") != frozen.get("test_files")):
        _fail("final generation receipt is absent, not exactly-once, or mismatched")
    access_path = root / "manifests/final_evaluation_access.json"
    access = _optional_object(access_path)
    if (access is None or access.get("status") != "DONE" or
            str(access.get("config_frozen_sha256", "")).upper() != sha256_file(frozen_path) or
            str(access.get("terminal_result_sha256", "")).upper() != sha256_file(done_path) or
            access.get("test_files") != frozen.get("test_files") or access.get("series") != "seeds_0_to_5"):
        _fail("confirmatory final access receipt is absent, incomplete, or mismatched")
    stats_path = root / "reports/confirmatory_statistics.json"
    if not stats_path.is_file() or str(done.get("statistics_sha256", "")).upper() != sha256_file(stats_path):
        _fail("confirmatory terminal result is not hash-bound to statistics")
    stats = load_json(stats_path)
    if not isinstance(stats, Mapping) or type(stats.get("positive_A")) is not bool or type(stats.get("positive_B")) is not bool:
        _fail("confirmatory statistics lack boolean positive_A/positive_B decisions")
    expected_result = "POSITIVE_CONFIRMED_A" if stats["positive_A"] else "NO_CONFIRMATORY_EVIDENCE_A"
    if (done.get("scientific_result") != expected_result or done.get("positive_A") != stats["positive_A"] or
            done.get("positive_B") != stats["positive_B"]):
        _fail("confirmatory terminal verdict differs from registered statistics")
    if str(done.get("config_frozen_sha256", "")).upper() != sha256_file(frozen_path):
        _fail("confirmatory terminal result CONFIG_FROZEN SHA-256 mismatch")
    return {
        "schema": "stage4.composition.v5.terminal-readiness.v1",
        "status": "PASS",
        "mode": "CONFIRM_COMPLETE",
        "terminal_path": done_path.relative_to(root).as_posix(),
        "scientific_result": done.get("scientific_result"),
        "pilot": pilot_inventory,
        "confirm_matrix": dict(confirm_matrix),
    }


def build_release(root: Path = ROOT, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    repo_root = Path(repo_root).resolve() if repo_root is not None else root.parents[1]
    atomic_manifest = find_atomic_reference_manifest(root)
    atomic = validate_atomic_reference_manifest(atomic_manifest, repo_root)
    atomic["path"] = atomic_manifest.relative_to(root).as_posix()
    syntax = validate_tree(root)
    atomic_root = sibling_atomic_root(root)
    atomic_syntax = validate_atomic_payload_syntax(atomic_root) if atomic_root is not None else None
    run_index = build_run_index(root, atomic_root)

    rankings = discover_ranking_files(root)
    ranking_reports = []
    if rankings:
        task_files = discover_task_files(root)
        tasks = load_task_index(task_files)
        for ranking in rankings:
            sibling = metrics_sibling(ranking)
            if not sibling.is_file():
                _fail(f"full ranking has no per-task metrics sibling: {ranking} -> {sibling}")
            receipt = receipt_sibling(sibling)
            if not receipt.is_file():
                _fail(f"full ranking has no DONE shard receipt: {ranking} -> {receipt}")
            ranking_report = validate_ranking_file(ranking, tasks, sibling, receipt)
            ranking_report["path"] = ranking.relative_to(root).as_posix()
            ranking_reports.append(ranking_report)
    pilot_decision = _optional_object(root / "runs/PILOT_DECISION.json")
    confirm_matrix = (validate_confirm_ranking_matrix(root, ranking_reports)
                      if pilot_decision is not None and pilot_decision.get("status") == "PASS" else None)
    terminal_readiness = validate_terminal_readiness(root, atomic, confirm_matrix)
    public_ranking_reports = [
        {key: value for key, value in report.items() if not key.startswith("_")}
        for report in ranking_reports
    ]
    ranking_validation = {
        "schema": "stage4.composition.v5.ranking-validation.v1",
        "ranking_files": public_ranking_reports,
        "ranking_file_count": len(public_ranking_reports),
        "ranking_rows": sum(item["rows"] for item in public_ranking_reports),
        "confirm_matrix": confirm_matrix,
    }
    registry = load_schema_registry(root / "schemas")
    validate_value(ranking_validation, registry[ranking_validation["schema"]], "ranking validation")
    validate_value(terminal_readiness, registry[terminal_readiness["schema"]], "terminal readiness")
    write_json(root / "manifests/ranking_validation.json", ranking_validation)
    write_json(root / "manifests/terminal_readiness.json", terminal_readiness)
    validation = {
        "schema": "stage4.composition.v5.validation.v1",
        "status": "PASS",
        "syntax_and_schema": syntax,
        "atomic_payload_syntax": atomic_syntax,
        "atomic_references": atomic,
        "run_count": run_index["run_count"],
        "ranking_file_count": ranking_validation["ranking_file_count"],
        "ranking_rows": ranking_validation["ranking_rows"],
        "terminal_readiness": terminal_readiness,
    }
    validate_value(validation, registry[validation["schema"]], "release validation")
    write_json(root / "manifests/VALIDATION.json", validation)
    receipt = create_release_archives(root)
    return {"validation": validation, "run_index": run_index, "archive_receipt": receipt}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fail-closed Stage 4 composition release validation")
    parser.add_argument("command", choices=("validate", "release"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate_tree(args.root)
    else:
        result = build_release(args.root, args.repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

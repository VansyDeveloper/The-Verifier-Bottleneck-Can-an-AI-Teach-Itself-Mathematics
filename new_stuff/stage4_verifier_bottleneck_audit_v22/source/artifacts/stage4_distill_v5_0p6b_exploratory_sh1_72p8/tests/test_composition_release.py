from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "code/composition_release.py"
SPEC = importlib.util.spec_from_file_location("composition_release", MODULE_PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R)


def strict_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def gzip_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def task_and_ranking(depth: int, *, seed: int = 0, branch: str = "composition_distill"):
    start = [1, 2, 3]
    field = 5
    witness = ("SH1",) * depth
    target = list(R.apply_program(start, witness, field))
    task = {
        "schema": "stage4.composition.v5.task.v1",
        "task_id": f"task-depth-{depth}",
        "p": field,
        "depth": depth,
        "start": start,
        "target": target,
    }
    entries = []
    for index, program in enumerate(R.enumerate_programs(depth)):
        entries.append({
            "program": list(program),
            "score": -float(index),
            "correct": R.apply_program(start, program, field) == tuple(target),
        })
    entries.sort(key=lambda row: (-row["score"], tuple(row["program"])))
    ranking = {
        "schema": "stage4.composition.v5.ranking.v1",
        "seed": seed,
        "branch": branch,
        "split": "final_a",
        "task_id": task["task_id"],
        "depth": depth,
        "p": field,
        "ranking": entries,
    }
    return task, ranking


def copy_schemas(root: Path) -> None:
    shutil.copytree(MODULE_PATH.parents[1] / "schemas", root / "schemas")


def make_atomic_reference(repo: Path, composition: Path, *, status: str = "PASS"):
    atomic_root = repo / "artifacts/stage4_sh1_v5_0p6b"
    decision = atomic_root / "runs/ATOMIC_DECISION.json"
    strict_dump(decision, {"status": status})
    paths = [decision]
    selected = None
    if status == "PASS":
        adapter = atomic_root / "adapters/selected"
        receipt = adapter / "training_receipt.json"
        strict_dump(receipt, {"status": "DONE"})
        (adapter / "adapter.safetensors").write_bytes(b"atomic-adapter")
        adapter_files = sorted(path for path in adapter.rglob("*") if path.is_file())
        paths.extend(adapter_files)
        selected = {
            "path": adapter.relative_to(repo).as_posix(),
            "tree_sha256": R._tree_sha256(adapter),
            "receipt_path": receipt.relative_to(repo).as_posix(),
            "receipt_sha256": R.sha256_file(receipt),
            "files_count": len(adapter_files),
            "bytes": sum(path.stat().st_size for path in adapter_files),
        }
    files = [{
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": R.sha256_file(path),
    } for path in sorted(set(paths))]
    manifest = {"schema": "stage4.composition.v5.atomic-references.v1", "files": files}
    if selected is not None:
        manifest["selected_adapter"] = selected
    manifest_path = composition / "manifests/atomic_reference.json"
    strict_dump(manifest_path, manifest)
    return manifest_path, manifest


def test_strict_json_and_gzip_jsonl_fail_closed(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(R.ReleaseValidationError, match="duplicate JSON key"):
        R.load_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x": NaN}\n', encoding="utf-8")
    with pytest.raises(R.ReleaseValidationError, match="non-finite"):
        R.load_json(nonfinite)

    stream = tmp_path / "rows.jsonl.gz"
    gzip_rows(stream, [{"x": 1}, {"x": 2}])
    assert R.load_jsonl(stream) == [{"x": 1}, {"x": 2}]

    blank = tmp_path / "blank.jsonl.gz"
    with gzip.open(blank, "wt", encoding="utf-8") as output:
        output.write('{"x": 1}\n\n')
    with pytest.raises(R.ReleaseValidationError, match="blank JSONL"):
        R.load_jsonl(blank)


@pytest.mark.parametrize("depth,expected", [(2, 25), (3, 125), (4, 625)])
def test_full_ranking_recomputation_all_registered_depths(depth: int, expected: int):
    task, ranking = task_and_ranking(depth)
    result = R.validate_ranking_record(ranking, task)
    assert len(ranking["ranking"]) == expected
    assert result["best_rank"] >= 1
    assert result["mrr"] == 1.0 / result["best_rank"]
    assert set(result) >= {"hit@1", "hit@8", "hit@16", "hit@32", "hit@64", "correct_mass", "log_gap"}


def test_full_ranking_rejects_missing_duplicate_bad_order_and_bad_label():
    task, ranking = task_and_ranking(2)

    missing = copy.deepcopy(ranking)
    missing["ranking"].pop()
    with pytest.raises(R.ReleaseValidationError, match="size mismatch"):
        R.validate_ranking_record(missing, task)

    duplicate = copy.deepcopy(ranking)
    duplicate["ranking"][-1] = copy.deepcopy(duplicate["ranking"][0])
    with pytest.raises(R.ReleaseValidationError, match="exact unique"):
        R.validate_ranking_record(duplicate, task)

    unordered = copy.deepcopy(ranking)
    unordered["ranking"][0], unordered["ranking"][1] = unordered["ranking"][1], unordered["ranking"][0]
    with pytest.raises(R.ReleaseValidationError, match="ranking order"):
        R.validate_ranking_record(unordered, task)

    mislabeled = copy.deepcopy(ranking)
    mislabeled["ranking"][0]["correct"] = not mislabeled["ranking"][0]["correct"]
    with pytest.raises(R.ReleaseValidationError, match="incorrect exact label"):
        R.validate_ranking_record(mislabeled, task)


def test_gzip_ranking_matches_complete_reported_task_metrics(tmp_path: Path):
    task, ranking = task_and_ranking(3)
    recomputed = R.validate_ranking_record(ranking, task)
    ranking_path = tmp_path / "seed0.rankings.jsonl.gz"
    metrics_path = tmp_path / "seed0.metrics.jsonl"
    gzip_rows(ranking_path, [ranking])
    metric = {
        "schema": "stage4.composition.v5.task-metrics.v1",
        "seed": ranking["seed"],
        "branch": ranking["branch"],
        "split": ranking["split"],
        "task_id": ranking["task_id"],
        "depth": ranking["depth"],
        "p": ranking["p"],
        **{key: value for key, value in recomputed.items() if key not in {"task_id", "depth"}},
    }
    strict_dump(metrics_path, metric)
    report = R.validate_ranking_file(ranking_path, {task["task_id"]: task}, metrics_path)
    assert report["rows"] == 1
    assert report["depths"] == {"3": 1}
    assert report["means"]["hit@32"] == recomputed["hit@32"]

    metric["mrr"] += 0.1
    strict_dump(metrics_path, metric)
    with pytest.raises(R.ReleaseValidationError, match="metric mismatch"):
        R.validate_ranking_file(ranking_path, {task["task_id"]: task}, metrics_path)


def test_current_eval_shard_shape_binding_and_schema_are_supported(tmp_path: Path):
    copy_schemas(tmp_path)
    task, ranking = task_and_ranking(2)
    fingerprint = "A" * 64
    task["task_fingerprint"] = fingerprint
    ranking.pop("seed")
    ranking.update({
        "schema": "stage4.distill.v5.ranking.v1",
        "task_fingerprint": fingerprint,
        "start": task["start"],
        "target": task["target"],
        "binding": {"seed": 4, "checkpoint_sha256": "B" * 64},
    })
    for rank, entry in enumerate(ranking["ranking"], 1):
        entry["rank"] = rank
    recomputed = R.validate_ranking_record(ranking, task)
    R.validate_value(ranking, R.load_json(tmp_path / "schemas/ranking.schema.json"), "ranking")

    ranking_path = tmp_path / "runs/confirm_seed4/rankings/composition_distill/part-00000.jsonl.gz"
    metric_path = tmp_path / "runs/confirm_seed4/metrics/composition_distill/part-00000.jsonl"
    gzip_rows(ranking_path, [ranking])
    metric = {
        "schema": "stage4.distill.v5.task-metrics.v1",
        "task_id": task["task_id"],
        "task_fingerprint": fingerprint,
        "split": "final_a",
        "p": task["p"],
        "depth": task["depth"],
        "branch": "composition_distill",
        "binding": ranking["binding"],
        **{key: value for key, value in recomputed.items() if key not in {"task_id", "depth"}},
    }
    strict_dump(metric_path, metric)
    receipt_path = R.receipt_sibling(metric_path)
    strict_dump(receipt_path, {
        "schema": "stage4.distill.v5.eval-shard-receipt.v1",
        "status": "DONE",
        "branch": "composition_distill",
        "binding": ranking["binding"],
        "task_ids": [task["task_id"]],
        "ranking_sha256": R.sha256_file(ranking_path).lower(),
        "metrics_sha256": R.sha256_file(metric_path).lower(),
        "ranking_rows": 1,
        "metrics_rows": 1,
    })
    R.validate_value(metric, R.load_json(tmp_path / "schemas/task_metrics.schema.json"), "metric")
    assert R.metrics_sibling(ranking_path) == metric_path
    assert R.discover_ranking_files(tmp_path) == [ranking_path]
    assert R.validate_ranking_file(ranking_path, {task["task_id"]: task}, metric_path, receipt_path)["rows"] == 1
    bad_receipt = R.load_json(receipt_path); bad_receipt["ranking_sha256"] = "0" * 64; strict_dump(receipt_path, bad_receipt)
    with pytest.raises(R.ReleaseValidationError, match="ranking SHA-256 mismatch"):
        R.validate_ranking_file(ranking_path, {task["task_id"]: task}, metric_path, receipt_path)


def test_task_schema_accepts_hash_bound_confirm_atomic_rows(tmp_path: Path):
    copy_schemas(tmp_path)
    start = [1, 2, 3]
    target = list(R.apply_program(start, ["SH1"], 5))
    row = {
        "schema": "stage4.composition.v5.task.v1",
        "task_id": "confirm-atomic-1",
        "task_fingerprint": "C" * 64,
        "split": "confirm_atomic",
        "family": "ATOMIC",
        "p": 5,
        "degree": 2,
        "depth": 1,
        "operation": "SH1",
        "start": start,
        "target": target,
        "witness": ["SH1"],
        "states": [start, target],
        "motif_count": 0,
        "shortest_depth": 1,
        "shortest_solution_count": 1,
        "solutions": [{"program": ["SH1"], "states": [start, target]}],
    }
    schema = R.load_json(tmp_path / "schemas/task.schema.json")
    R.validate_value(row, schema, "confirm atomic row")
    bad = dict(row); bad.pop("operation")
    with pytest.raises(R.ReleaseValidationError, match="schema validation"):
        R.validate_value(bad, schema, "bad confirm atomic row")


def test_run_index_preserves_arbitrary_scientific_outcomes(tmp_path: Path):
    copy_schemas(tmp_path)
    output = tmp_path / "manifests/RUN_INDEX.json"
    index = R.write_run_index([
        {"run_id": "seed0-control", "status": "DONE", "branch": "atomic_control"},
        {"run_id": "seed0-distill", "status": "FAILED_CALIBRATION_GATE", "branch": "composition_distill"},
    ], output)
    assert index["status_counts"] == {"DONE": 1, "FAILED_CALIBRATION_GATE": 1}
    assert [row["run_id"] for row in index["runs"]] == ["seed0-control", "seed0-distill"]
    assert index["pilot_attempts"] == []
    assert index["pilot_decision"] is None
    assert index["terminal"] is None
    assert index["status_record_count"] == 0
    assert R.validate_json_file(output, tmp_path / "schemas/run_index.schema.json") == index


def test_unknown_schema_id_fails_only_inside_critical_release_paths(tmp_path: Path):
    copy_schemas(tmp_path)
    strict_dump(tmp_path / "reports/unknown.json", {"schema": "stage4.unknown.v999", "value": 1})
    with pytest.raises(R.ReleaseValidationError, match="unknown schema id in critical artifact"):
        R.validate_tree(tmp_path)
    (tmp_path / "reports/unknown.json").unlink()
    strict_dump(tmp_path / "scratch.json", {"schema": "unregistered.scratch", "value": 1})
    result = R.validate_tree(tmp_path)
    assert result["syntax_only_documents"] == 1


def test_discovered_run_index_exposes_pilot_status_and_authoritative_terminal(tmp_path: Path):
    copy_schemas(tmp_path)
    strict_dump(tmp_path / "runs/pilot/p0/attempt.json", {
        "schema": "stage4.composition.v5.pilot-attempt.v1", "status": "DONE",
        "attempt_id": "p0", "resolved_config": {}, "checks": {}, "pass": False,
    })
    strict_dump(tmp_path / "runs/PILOT_DECISION.json", {
        "schema": "stage4.composition.v5.pilot-decision.v1", "status": "FAILED_PILOT_GATE",
        "model": "Qwen/Qwen3-0.6B", "selected_attempt": "p0", "selected": {}, "attempts": ["p0"],
        "final_created": False, "composition_confirm_unlocked": False,
    })
    strict_dump(tmp_path / "runs/STAGE4_FAILED.json", {
        "schema": "stage4.composition.v5.terminal.v1", "status": "FAILED", "phase": "pilot_gate",
        "scientific_result": "no_confirmatory_evidence",
    })
    strict_dump(tmp_path / "runs/training/example.status.json", {
        "schema": "stage4.distill.v5.training-status.v1", "status": "DONE", "branch": "atomic_control",
        "seed": 0, "effective_batch": 64, "epochs": 2, "input_sha256": "A" * 64,
    })
    index = R.build_run_index(tmp_path)
    assert index["pilot_attempt_count"] == 1
    assert index["pilot_decision"]["status"] == "FAILED_PILOT_GATE"
    assert index["status_record_count"] == 1
    assert index["terminal"]["scientific_result"] == "no_confirmatory_evidence"
    assert index["kind_counts"]["terminal_result"] == 1


def test_confirm_matrix_requires_every_frozen_split_seed_and_branch(tmp_path: Path):
    root = tmp_path / "stage4_distill_v5_0p6b"
    test_files = {}
    task_ids = {}
    for split in R.CONFIRM_FROZEN_SPLITS:
        path = root / "data" / f"{split}.jsonl"
        task_id = f"task::{split}"
        strict_dump(path, {"task_id": task_id, "split": split})
        test_files[path.name] = {"rows": 1, "bytes": path.stat().st_size, "sha256": R.sha256_file(path)}
        task_ids[split] = task_id
    atomic_path = root / "data/confirm_atomic.jsonl"
    strict_dump(atomic_path, {"task_id": "atomic", "split": "confirm_atomic"})
    test_files[atomic_path.name] = {"rows": 1, "bytes": atomic_path.stat().st_size, "sha256": R.sha256_file(atomic_path)}
    strict_dump(root / "CONFIG_SELECTION_FROZEN.json", {"schema": "stage4.composition.v5.selection-freeze.v1"})
    strict_dump(root / "configs/protocol.json", {"schema": "stage4.distill.v5.protocol.v1"})
    strict_dump(root / "manifests/final_data_manifest.json", {
        "schema": "stage4.composition.v5.data-manifest.v1", "status": "FROZEN", "files": test_files,
    })
    strict_dump(root / "CONFIG_FROZEN.json", {
        "schema": "stage4.composition.v5.config-freeze.v1", "status": "FROZEN",
        "model": "Qwen/Qwen3-0.6B", "selected_attempt": "pilot", "resolved_config": {},
        "confirm_seeds": list(R.CONFIRM_SEEDS), "test_files": test_files, "code": {},
        "selection_sha256": R.sha256_file(root / "CONFIG_SELECTION_FROZEN.json"),
        "protocol_sha256": R.sha256_file(root / "configs/protocol.json"),
    })
    frozen_sha = R.sha256_file(root / "CONFIG_FROZEN.json")
    reports = []
    for split in R.CONFIRM_FROZEN_SPLITS:
        for seed in R.CONFIRM_SEEDS:
            for branch in R.CONFIRM_BRANCHES:
                reports.append({
                    "path": f"runs/confirm/seed{seed}/{branch}/{split}/part.jsonl.gz",
                    "attempt": "confirm_frozen", "split": split,
                    "binding": {
                        "seed": seed, "attempt": "confirm_frozen", "branch": branch, "split": split,
                        "config_frozen_sha256": frozen_sha,
                        "data_sha256": test_files[f"{split}.jsonl"]["sha256"],
                    },
                    "_task_ids": [task_ids[split]],
                })
    matrix = R.validate_confirm_ranking_matrix(root, reports)
    assert matrix["cells"] == len(R.CONFIRM_FROZEN_SPLITS) * 6 * 3
    assert matrix["ranking_rows"] == matrix["cells"]
    with pytest.raises(R.ReleaseValidationError, match="incomplete confirm ranking matrix"):
        R.validate_confirm_ranking_matrix(root, reports[:-1])
    wrong_coverage = copy.deepcopy(reports)
    wrong_coverage[0]["_task_ids"] = ["not-a-frozen-task"]
    with pytest.raises(R.ReleaseValidationError, match="confirm task coverage mismatch"):
        R.validate_confirm_ranking_matrix(root, wrong_coverage)


def test_pilot_pass_cannot_be_released_before_confirmation(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "artifacts/stage4_distill_v5_0p6b"
    copy_schemas(root)
    strict_dump(root / "configs/protocol.json", {
        "schema": "stage4.distill.v5.protocol.v1", "stage": "Discover-and-Distill",
        "model": "Qwen/Qwen3-0.6B", "model_size_lock": "0.6B",
        "pilot_initial": {}, "confirm": {}, "ranking": {},
    })
    make_atomic_reference(repo, root, status="PASS")
    attempt = {
        "schema": "stage4.composition.v5.pilot-attempt.v1", "status": "DONE",
        "attempt_id": "pilot0", "resolved_config": {}, "checks": {}, "pass": True,
    }
    strict_dump(root / "runs/pilot/pilot0/attempt.json", attempt)
    strict_dump(root / "runs/PILOT_DECISION.json", {
        "schema": "stage4.composition.v5.pilot-decision.v1", "status": "PASS",
        "model": "Qwen/Qwen3-0.6B", "initial_attempt": "pilot0", "selected_attempt": "pilot0",
        "selected": attempt, "dev_cycle": {"used": False}, "attempts": ["pilot0"],
        "final_created": True, "composition_confirm_unlocked": True,
    })
    with pytest.raises(R.ReleaseValidationError, match="lacks CONFIG_FROZEN"):
        R.build_release(root, repo)


def test_confirm_terminal_requires_exactly_once_access_and_statistics_binding(tmp_path: Path):
    root = tmp_path / "stage4_distill_v5_0p6b"
    attempt = {
        "schema": "stage4.composition.v5.pilot-attempt.v1", "status": "DONE",
        "attempt_id": "p0", "resolved_config": {}, "checks": {}, "pass": True,
    }
    strict_dump(root / "runs/pilot/p0/attempt.json", attempt)
    test_files = {"confirm_atomic.jsonl": {"rows": 1, "bytes": 1, "sha256": "A" * 64}}
    strict_dump(root / "CONFIG_FROZEN.json", {"status": "FROZEN", "test_files": test_files})
    frozen_sha = R.sha256_file(root / "CONFIG_FROZEN.json")
    strict_dump(root / "runs/PILOT_DECISION.json", {
        "status": "PASS", "attempts": ["p0"], "selected_attempt": "p0", "selected": attempt,
        "final_created": True, "composition_confirm_unlocked": True, "config_frozen_sha256": frozen_sha,
    })
    stats_path = root / "reports/confirmatory_statistics.json"
    strict_dump(stats_path, {"positive_A": False, "positive_B": True})
    done_path = root / "runs/STAGE4_DONE.json"
    strict_dump(done_path, {
        "run_id": "stage4-confirmatory-series", "status": "DONE",
        "scientific_result": "NO_CONFIRMATORY_EVIDENCE_A", "positive_A": False, "positive_B": True,
        "statistics_sha256": R.sha256_file(stats_path), "config_frozen_sha256": frozen_sha,
    })
    generation_path = root / "manifests/final_generation_receipt.json"
    strict_dump(generation_path, {
        "status": "DONE", "evaluation_count": 1, "config_frozen_sha256": frozen_sha, "test_files": test_files,
    })
    strict_dump(root / "manifests/final_evaluation_access.json", {
        "status": "DONE", "config_frozen_sha256": frozen_sha, "test_files": test_files,
        "series": "seeds_0_to_5", "terminal_result_sha256": R.sha256_file(done_path),
    })
    atomic = {"atomic_decision": {"status": "PASS"}, "selected_adapter": {"path": "atomic"}}
    matrix = {"status": "PASS"}
    ready = R.validate_terminal_readiness(root, atomic, matrix)
    assert ready["mode"] == "CONFIRM_COMPLETE"
    generation = R.load_json(generation_path); generation["evaluation_count"] = 0; strict_dump(generation_path, generation)
    with pytest.raises(R.ReleaseValidationError, match="not exactly-once"):
        R.validate_terminal_readiness(root, atomic, matrix)


def test_atomic_reference_manifest_verifies_bytes_sha_and_rejects_archives(tmp_path: Path):
    repo = tmp_path / "repo"
    composition = repo / "artifacts/stage4_distill_v5_0p6b"
    copy_schemas(composition)
    manifest_path, manifest = make_atomic_reference(repo, composition)
    checked = R.validate_atomic_reference_manifest(manifest_path, repo)
    assert checked["atomic_decision"]["status"] == "PASS"
    assert checked["selected_adapter"]["tree_sha256"] == manifest["selected_adapter"]["tree_sha256"]

    missing_payload = copy.deepcopy(manifest)
    missing_payload["files"] = [item for item in missing_payload["files"] if not item["path"].endswith("adapter.safetensors")]
    strict_dump(manifest_path, missing_payload)
    with pytest.raises(R.ReleaseValidationError, match="payload references are incomplete"):
        R.validate_atomic_reference_manifest(manifest_path, repo)

    archive_item = copy.deepcopy(manifest["files"][0])
    archive_item["path"] = "artifacts/stage4_sh1_v5_0p6b/archives/old.zip"
    strict_dump(manifest_path, {"schema": "stage4.composition.v5.atomic-references.v1", "files": [archive_item]})
    with pytest.raises(R.ReleaseValidationError, match="old archives"):
        R.validate_atomic_reference_manifest(manifest_path, repo)


def test_compact_and_full_archives_have_exact_membership_and_crc(tmp_path: Path):
    root = tmp_path / "stage4_distill_v5_0p6b"
    copy_schemas(root)
    strict_dump(root / "configs/protocol.json", {"name": "test"})
    strict_dump(root / "reports/summary.json", {"result": "neutral"})
    strict_dump(root / "manifests/validation_input.json", {"ok": True})
    strict_dump(root / "rankings/final.summary.json", {"hit@32": 0.5})
    (root / "rankings/final.rankings.jsonl.gz").parent.mkdir(parents=True, exist_ok=True)
    (root / "rankings/final.rankings.jsonl.gz").write_bytes(b"raw-ranking-placeholder")
    (root / "adapters/seed0").mkdir(parents=True)
    (root / "adapters/seed0/adapter.safetensors").write_bytes(b"adapter")

    assert set(R.archive_member_lists(root)) == {"compact", "full"}
    receipt = R.create_release_archives(root)
    assert receipt["atomic_payload_manifest"] is None
    compact_path = root / receipt["archives"]["compact"]["path"]
    full_path = root / receipt["archives"]["full"]["path"]
    for path in (compact_path, full_path):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            kind = "compact" if path == compact_path else "full"
            assert receipt["archives"][kind]["membership_sha256"] == R._archive_membership_sha256(archive.namelist())
            assert receipt["archives"][kind]["roots"] == {root.name: len(archive.namelist())}
    with zipfile.ZipFile(compact_path) as archive:
        names = archive.namelist()
        assert any(name.endswith("reports/summary.json") for name in names)
        assert any(name.endswith("rankings/final.summary.json") for name in names)
        assert not any("adapters/" in name for name in names)
        assert not any(name.endswith(".rankings.jsonl.gz") for name in names)
    with zipfile.ZipFile(full_path) as archive:
        names = archive.namelist()
        assert any("adapters/seed0/adapter.safetensors" in name for name in names)
        assert any(name.endswith("rankings/final.rankings.jsonl.gz") for name in names)
    assert R.load_json(root / "manifests/archive_receipt.json") == receipt
    assert R.load_json(root / "archives/RELEASE.json") == receipt


def test_full_archive_includes_hash_bound_sibling_atomic_payload(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "artifacts/stage4_distill_v5_0p6b"
    atomic_root = repo / "artifacts/stage4_sh1_v5_0p6b"
    copy_schemas(root)
    strict_dump(root / "configs/protocol.json", {"name": "composition"})
    strict_dump(root / "reports/summary.json", {"verdict": "complete"})

    expected_payload = {
        "code/v5_train.py": b"print('atomic')\n",
        "configs/protocol.json": b'{"model":"Qwen3-0.6B"}\n',
        "data/train.jsonl": b'{"task_id":"atomic-1"}\n',
        "manifests/data.json": b'{"status":"FROZEN"}\n',
        "runs/ATOMIC_DECISION.json": b'{"status":"PASS"}\n',
        "logs/train.log": b"finished\n",
        "adapters/selected/adapter.safetensors": b"atomic-adapter",
        "archives/README.md": b"zip files are intentionally omitted\n",
    }
    for rel, payload in expected_payload.items():
        path = atomic_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    excluded_payload = {
        "__pycache__/v5_train.pyc": b"cache",
        ".cache/model.bin": b"cache",
        "runs/interrupted.partial/state.json": b"partial",
        "tmp/work.json": b"temporary",
        "logs/latest.tmp": b"temporary",
        "archives/old.zip": b"old-archive",
    }
    for rel, payload in excluded_payload.items():
        path = atomic_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    receipt = R.create_release_archives(root)
    binding = receipt["atomic_payload_manifest"]
    assert binding is not None
    compact_path = root / receipt["archives"]["compact"]["path"]
    full_path = root / receipt["archives"]["full"]["path"]
    with zipfile.ZipFile(compact_path) as archive:
        compact_names = archive.namelist()
        assert archive.testzip() is None
        archived_atomic_compact = {
            name.removeprefix(f"{atomic_root.name}/")
            for name in compact_names
            if name.startswith(f"{atomic_root.name}/")
        }
        assert archived_atomic_compact == {
            "code/v5_train.py", "configs/protocol.json", "manifests/data.json", "runs/ATOMIC_DECISION.json"
        }
    with zipfile.ZipFile(full_path) as archive:
        full_names = archive.namelist()
        assert archive.testzip() is None
        archived_atomic = {
            name.removeprefix(f"{atomic_root.name}/")
            for name in full_names
            if name.startswith(f"{atomic_root.name}/")
        }
        assert archived_atomic == set(expected_payload)
        assert not any(rel in archived_atomic for rel in excluded_payload)
        assert receipt["archives"]["full"]["membership_sha256"] == R._archive_membership_sha256(full_names)
        assert receipt["archives"]["full"]["roots"][atomic_root.name] == len(expected_payload)
        assert receipt["archives"]["full"]["roots"][root.name] == len(full_names) - len(expected_payload)

    manifest_path = root / R.ATOMIC_PAYLOAD_MANIFEST.as_posix()
    manifest = R.load_json(manifest_path)
    assert set(manifest["files"]) == set(expected_payload)
    assert binding == {
        "path": R.ATOMIC_PAYLOAD_MANIFEST.as_posix(),
        "sha256": R.sha256_file(manifest_path),
        "root": atomic_root.name,
        "tree_sha256": manifest["tree_sha256"],
        "file_count": len(expected_payload),
        "total_bytes": sum(len(payload) for payload in expected_payload.values()),
    }
    assert R.validate_atomic_payload_manifest(root, atomic_root) == manifest
    archive_contents = R.load_json(root / "manifests/archive_contents.json")
    assert set(archive_contents["atomic_full"]) == set(expected_payload)
    assert set(archive_contents["atomic_compact"]) == {
        "code/v5_train.py", "configs/protocol.json", "manifests/data.json", "runs/ATOMIC_DECISION.json"
    }

    (atomic_root / "code/v5_train.py").write_bytes(b"tampered\n")
    with pytest.raises(R.ReleaseValidationError, match="atomic payload SHA-256 mismatch"):
        R.validate_atomic_payload_manifest(root, atomic_root)


def test_sha_manifest_excludes_archives_self_hash_and_old_zip(tmp_path: Path):
    root = tmp_path / "stage4_distill_v5_0p6b"
    copy_schemas(root)
    strict_dump(root / "reports/report.json", {"ok": True})
    (root / "archives").mkdir(parents=True)
    (root / "archives/old.zip").write_bytes(b"old")
    manifest = R.build_sha256_manifest(root)
    assert "reports/report.json" in manifest["files"]
    assert "manifests/SHA256SUMS.json" not in manifest["files"]
    assert not any(path.endswith(".zip") for path in manifest["files"])
    assert manifest["files"]["reports/report.json"]["sha256"] == R.sha256_file(root / "reports/report.json")
    R.validate_sha256_manifest(root)
    strict_dump(root / "reports/report.json", {"ok": False})
    with pytest.raises(R.ReleaseValidationError, match="SHA-256 manifest mismatch"):
        R.validate_sha256_manifest(root)


def test_end_to_end_atomic_negative_release_is_terminal_and_archivable(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "artifacts/stage4_distill_v5_0p6b"
    copy_schemas(root)
    strict_dump(root / "configs/protocol.json", {
        "schema": "stage4.distill.v5.protocol.v1", "stage": "Discover-and-Distill",
        "model": "Qwen/Qwen3-0.6B", "model_size_lock": "0.6B",
        "pilot_initial": {}, "confirm": {}, "ranking": {},
    })
    strict_dump(root / "reports/summary.json", {"verdict": "derived elsewhere"})
    make_atomic_reference(repo, root, status="FAILED_CALIBRATION_GATE")
    decision_path = repo / "artifacts/stage4_sh1_v5_0p6b/runs/ATOMIC_DECISION.json"
    strict_dump(root / "runs/STAGE4_DONE.json", {
        "schema": "stage4.composition.v5.terminal.v1", "run_id": "stage4-atomic-negative",
        "status": "DONE", "scientific_result": "FAILED_ATOMIC_GATE",
        "composition_started": False, "final_composition_created": False,
        "atomic_decision_sha256": R.sha256_file(decision_path),
    })
    result = R.build_release(root, repo)
    assert result["validation"]["status"] == "PASS"
    assert result["validation"]["terminal_readiness"]["mode"] == "ATOMIC_NEGATIVE"
    assert result["validation"]["ranking_rows"] == 0
    assert result["run_index"]["run_count"] == 2
    assert result["run_index"]["terminal"]["scientific_result"] == "FAILED_ATOMIC_GATE"
    assert any(row["run_id"] == "atomic::decision" and row["status"] == "FAILED_CALIBRATION_GATE"
               for row in result["run_index"]["runs"])
    assert (root / "archives/stage4_distill_v5_0p6b_audit_report.zip").is_file()
    assert (root / "archives/stage4_distill_v5_0p6b_complete.zip").is_file()


def test_atomic_negative_terminal_rejects_composition_runtime_payload(tmp_path: Path):
    root = tmp_path / "stage4_distill_v5_0p6b"
    strict_dump(root / "runs/STAGE4_DONE.json", {
        "run_id": "stage4-atomic-negative", "status": "DONE", "scientific_result": "FAILED_ATOMIC_GATE",
        "composition_started": False, "final_composition_created": False, "atomic_decision_sha256": "ABCD",
    })
    atomic = {"atomic_decision": {"status": "FAILED_CALIBRATION_GATE", "sha256": "ABCD"}}
    (root / "data").mkdir(parents=True); (root / "data/leak.jsonl").write_text('{"task_id":"leak"}\n', encoding="utf-8")
    with pytest.raises(R.ReleaseValidationError, match="composition runtime payload"):
        R.validate_terminal_readiness(root, atomic, None)

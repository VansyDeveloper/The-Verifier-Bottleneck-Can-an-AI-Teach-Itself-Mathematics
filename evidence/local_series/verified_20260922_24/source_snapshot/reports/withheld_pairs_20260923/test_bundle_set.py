import json
import hashlib
import zipfile
from pathlib import Path

import pytest

from bundle_set import (build_bundle, candidates, checkpoint_inventory, require_evidence,
                        validate_docker_inspect, write_archive)


def test_candidates_include_requested_raw_evidence_but_not_other_set_or_weights(tmp_path):
    def add(relative: str) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"evidence")
        return path

    selected = {
        add("artifacts/withheld_pairs_20260923/FREEZE.json"),
        add("artifacts/withheld_pairs_20260923/adapters/k1_s2_seed85000_random/training_receipt.json"),
        add("artifacts/withheld_pairs_20260923/rankings/rankings/k1_s2_seed85000_random_withheld/part-00000.jsonl.gz"),
        add("artifacts/withheld_pairs_20260923/runs/docker/verifier_withheld_eval_20260923_k1s2_85000_random.stdout.log"),
        add("artifacts/withheld_pairs_20260923/runs/k1_s2_allocation.json"),
        add("artifacts/withheld_pairs_20260923/runs/queue_k1s2.restart1.stdout.log"),
        add("reports/withheld_pairs_20260923/data/k1_s2/analysis_summary.json"),
        add("reports/trajectory_diversity_20260922/WITHHELD_PAIR_SETS.json"),
        add("artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_core.py"),
    }
    add("artifacts/withheld_pairs_20260923/adapters/k1_s2_seed85000_random/adapter_model.safetensors")
    add("artifacts/withheld_pairs_20260923/rankings/rankings/k1_s3_seed85000_random_withheld/part-00000.jsonl.gz")

    assert set(candidates(tmp_path, 1, 2)) == selected


def test_missing_docker_stderr_prevents_claiming_a_complete_set(tmp_path):
    artifact = tmp_path / "artifacts/withheld_pairs_20260923"
    report = tmp_path / "reports/withheld_pairs_20260923/data/k1_s2"
    report.mkdir(parents=True)
    (report / "analysis_summary.json").write_text(json.dumps({
        "status": "VALID", "k": 1, "subset": 2,
        "completed_trainings": 6, "completed_evaluations": 6,
    }), encoding="utf-8")
    runs = artifact / "runs"
    logs = runs / "docker"
    logs.mkdir(parents=True)
    (runs / "queue_k1s2.jsonl").write_text('{"status":"SET_DONE"}\n', encoding="utf-8")
    for mode in ("train", "eval"):
        for seed in (85000, 85001, 85002):
            for arm in ("random", "withheld"):
                stem = f"verifier_withheld_{mode}_20260923_k1s2_{seed}_{arm}"
                for suffix in ("inspect.json", "stdout.log", "stderr.log"):
                    (logs / f"{stem}.{suffix}").write_text("evidence", encoding="utf-8")
    (logs / "verifier_withheld_eval_20260923_k1s2_85002_withheld.stderr.log").unlink()

    with pytest.raises(ValueError, match="Docker evidence missing"):
        require_evidence(tmp_path, 1, 2)


def test_archive_contains_verified_manifest_and_only_selected_files(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "reports/withheld_pairs_20260923/data/k1_s2/analysis_summary.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"status":"VALID"}\n')
    other = repo / "artifacts/withheld_pairs_20260923/adapters/k1_s2_seed85000_random/adapter_model.safetensors"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"excluded weights")
    target = tmp_path / "bundle.zip"
    inventory = [{"seed": 85000, "arm": "random", "weights_sha256": hashlib.sha256(other.read_bytes()).hexdigest().upper(),
                  "weights_bytes": other.stat().st_size, "weights_in_audit_zip": False}]

    write_archive(repo, target, [source], inventory)

    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "reports/withheld_pairs_20260923/data/k1_s2/analysis_summary.json",
            "AUDIT_MANIFEST.json", "CHECKPOINT_INVENTORY.json",
        }
        manifest = json.loads(archive.read("AUDIT_MANIFEST.json"))
        assert manifest == [{"path": source.relative_to(repo).as_posix(),
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper(),
                             "bytes": source.stat().st_size}]
        assert json.loads(archive.read("CHECKPOINT_INVENTORY.json")) == inventory
    assert target.with_suffix(".zip.sha256").read_text().strip() == (
        f"{hashlib.sha256(target.read_bytes()).hexdigest().upper()}  {target.name}")


def test_checkpoint_inventory_requires_all_six_weights(tmp_path):
    with pytest.raises(ValueError, match="checkpoint missing"):
        checkpoint_inventory(tmp_path, 1, 2)


def test_build_bundle_refuses_incomplete_set_before_writing(tmp_path):
    target = tmp_path / "bundle.zip"
    with pytest.raises((FileNotFoundError, ValueError)):
        build_bundle(tmp_path, 1, 2, target)
    assert not target.exists()


def test_docker_inspect_must_match_assigned_gpu_and_exit_zero():
    inspect = {"State": {"Status": "exited", "ExitCode": 0},
               "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["0"]}]}}
    validate_docker_inspect(inspect, 0)
    with pytest.raises(ValueError, match="GPU binding"):
        validate_docker_inspect(inspect, 1)
    inspect["State"]["ExitCode"] = 1
    with pytest.raises(ValueError, match="exit state"):
        validate_docker_inspect(inspect, 0)

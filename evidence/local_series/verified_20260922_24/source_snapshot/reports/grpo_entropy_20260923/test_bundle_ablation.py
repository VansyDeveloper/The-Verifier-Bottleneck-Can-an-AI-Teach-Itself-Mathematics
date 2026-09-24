import json
import hashlib
import zipfile
from pathlib import Path

import pytest

from bundle_ablation import (candidates, checkpoint_inventory, require_complete,
                            validate_binding, write_archive)


def test_candidates_keep_new_grpo_evidence_but_exclude_weights_and_other_studies(tmp_path):
    def add(relative: str) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"evidence")
        return path

    chosen = {
        add("reports/grpo_entropy_20260923/PROTOCOL.md"),
        add("reports/grpo_entropy_20260923/data/paired_ablation/analysis_summary.json"),
        add("artifacts/grpo_entropy_20260923/holdout_1000.jsonl"),
        add("artifacts/grpo_entropy_20260923/runs/seed0_control_fp32/metrics.jsonl"),
        add("artifacts/grpo_entropy_20260923/runs/seed0_control_fp32/evaluation/shards/part-00000.jsonl.gz"),
        add("artifacts/grpo_entropy_20260923/queue_evidence/job.inspect.json"),
        add("artifacts/data/pilot/rl_train.jsonl"),
        add("scripts/train_action_grpo.py"),
        add("src/vbexp/verifier.py"),
    }
    add("artifacts/grpo_entropy_20260923/runs/seed0_control_fp32/final_adapter/adapter_model.safetensors")
    add("artifacts/withheld_pairs_20260923/rankings/other.jsonl.gz")
    add("reports/grpo_entropy_20260923/data/paired_ablation/paired_audit.zip")

    assert set(candidates(tmp_path)) == chosen


def test_complete_bundle_requires_full_independent_audit(tmp_path):
    out = tmp_path / "reports/grpo_entropy_20260923/data/paired_ablation"
    out.mkdir(parents=True)
    (out / "analysis_summary.json").write_text(json.dumps({
        "status": "VALID", "completed_trainings": 6,
        "completed_evaluations": 5,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="independent audit"):
        require_complete(tmp_path)


def test_valid_summary_does_not_replace_missing_queue_receipts(tmp_path):
    out = tmp_path / "reports/grpo_entropy_20260923/data/paired_ablation"
    out.mkdir(parents=True)
    (out / "analysis_summary.json").write_text(json.dumps({
        "status": "VALID", "completed_trainings": 6,
        "completed_evaluations": 6, "candidate_program_scores": 750000,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="queue"):
        require_complete(tmp_path)


def test_complete_queues_still_require_all_twelve_docker_inspections(tmp_path):
    out = tmp_path / "reports/grpo_entropy_20260923/data/paired_ablation"
    out.mkdir(parents=True)
    (out / "analysis_summary.json").write_text(json.dumps({
        "status": "VALID", "completed_trainings": 6,
        "completed_evaluations": 6, "candidate_program_scores": 750000,
    }), encoding="utf-8")
    artifact = tmp_path / "artifacts/grpo_entropy_20260923"
    artifact.mkdir(parents=True)
    for gpu_index in (1, 2):
        (artifact / f"queue_gpu{gpu_index}_fp32.jsonl").write_text('{"status":"QUEUE_DONE"}\n')

    with pytest.raises(ValueError, match="Docker evidence"):
        require_complete(tmp_path)


def test_checkpoint_inventory_requires_six_final_adapters(tmp_path):
    with pytest.raises(ValueError, match="final adapter missing"):
        checkpoint_inventory(tmp_path)


def test_archive_has_file_hash_manifest_and_checkpoint_inventory(tmp_path):
    source = tmp_path / "repo/reports/grpo_entropy_20260923/data/paired_ablation/analysis_summary.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"status":"VALID"}\n')
    target = tmp_path / "paired_audit.zip"
    inventory = [{"seed": 0, "arm": "control", "weights_in_audit_zip": False}]

    write_archive(tmp_path / "repo", target, [source], inventory)

    with zipfile.ZipFile(target) as handle:
        assert handle.testzip() is None
        assert set(handle.namelist()) == {
            source.relative_to(tmp_path / "repo").as_posix(),
            "AUDIT_MANIFEST.json", "CHECKPOINT_INVENTORY.json",
        }
        assert json.loads(handle.read("AUDIT_MANIFEST.json")) == [{
            "path": source.relative_to(tmp_path / "repo").as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper(),
            "bytes": source.stat().st_size,
        }]
        assert json.loads(handle.read("CHECKPOINT_INVENTORY.json")) == inventory
    assert target.with_suffix(".zip.sha256").read_text().strip() == (
        f"{hashlib.sha256(target.read_bytes()).hexdigest().upper()}  {target.name}")


def test_frozen_holdout_hash_is_rechecked_after_independent_audit(tmp_path):
    holdout = tmp_path / "artifacts/grpo_entropy_20260923/holdout_1000.jsonl"
    holdout.parent.mkdir(parents=True)
    holdout.write_bytes(b"frozen")
    duration = tmp_path / "reports/grpo_entropy_20260923/DURATION_FREEZE_FP32.json"
    duration.parent.mkdir(parents=True)
    duration.write_bytes(b"duration")
    summary = {
        "holdout_sha256": hashlib.sha256(holdout.read_bytes()).hexdigest().upper(),
        "duration_freeze_sha256": hashlib.sha256(duration.read_bytes()).hexdigest().upper(),
    }
    validate_binding(tmp_path, summary)
    holdout.write_bytes(b"changed")
    with pytest.raises(ValueError, match="holdout SHA"):
        validate_binding(tmp_path, summary)

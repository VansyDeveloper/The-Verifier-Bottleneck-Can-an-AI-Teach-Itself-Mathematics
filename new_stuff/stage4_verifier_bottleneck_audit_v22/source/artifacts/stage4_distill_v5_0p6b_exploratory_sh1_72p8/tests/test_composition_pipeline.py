import copy
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import composition_core as core
import composition_eval as evaluation
import composition_model as model_code
STAGE4_SPEC = importlib.util.spec_from_file_location("stage4_composition_v5", ROOT / "code/stage4.py")
stage4 = importlib.util.module_from_spec(STAGE4_SPEC); STAGE4_SPEC.loader.exec_module(stage4)


def encoded(target_tokens, length, kind="atomic_plan", operation="SH1"):
    labels = [-100] * (length - target_tokens) + list(range(1, target_tokens + 1))
    return {"input_ids": list(range(10, 10 + length)), "labels": labels,
            "position_ids": list(range(length)), "target_tokens": target_tokens,
            "prompt_tokens": length - target_tokens, "total_tokens": length,
            "kind": kind, "operation": operation, "segments": 1}


def test_protocol_is_hard_locked_to_0p6b():
    stage4.protocol_guard()
    assert stage4.PROTOCOL["model"] == "Qwen/Qwen3-0.6B"
    assert "1.7B" not in str(stage4.PROTOCOL)
    assert stage4.PROTOCOL["registered_stage4_result_remains"] == "FAILED_CALIBRATION_GATE"
    assert stage4.PROTOCOL["numeric_runtime"] == {
        "atomic_source_evaluation_dtype": "float32",
        "atomic_export_storage_dtype": "float32",
        "branch_master_dtype": "float32",
        "branch_compute_dtype": "bfloat16_autocast",
        "reason": "preserve the frozen corrective evaluator function while retaining preregistered bf16 training compute",
    }


def test_scientific_training_refuses_cpu_or_non_bf16_fallback(monkeypatch):
    resolved = stage4._resolved_config(seed=0, lr=1e-4, epochs=2, replay=0.2)
    monkeypatch.setattr(model_code.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        model_code.require_bf16_cuda_training(resolved)
    monkeypatch.setattr(model_code.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_code.torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="bf16"):
        model_code.require_bf16_cuda_training(resolved)


def test_control_packing_matches_every_distill_target_budget():
    control = [encoded(2, 5, "atomic_plan", "SH1"), encoded(4, 8, "atomic_apply", "SC2")]
    distill = [encoded(3, 7, "composition_distill"), encoded(5, 9, "composition_distill"),
               encoded(8, 12, "composition_distill")]
    packed, unchanged, manifest = model_code.pack_control_to_budget(control, distill, seed=3, max_length=64)
    assert unchanged is distill
    assert [row["target_tokens"] for row in packed] == [3, 5, 8]
    assert manifest["loss_bearing_target_tokens_each"] == 16
    assert manifest["per_record_target_budget_equal"] is True
    assert any(row["segments"] > 1 for row in packed)
    for row in packed:
        starts = [index for index, position in enumerate(row["position_ids"]) if position == 0]
        assert len(starts) == row["segments"]


def test_packed_collate_keeps_reset_positions_and_isolated_padding():
    class Tokenizer:
        pad_token_id = 0
    batch = model_code.collate_packed(Tokenizer(), [
        {**encoded(2, 5), "position_ids": [0, 1, 2, 0, 1]},
        encoded(2, 3),
    ])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 0, 1]
    # Two ignored pad tokens form a separate prefix segment; the real sequence resets to zero.
    assert batch["position_ids"][1].tolist() == [0, 1, 0, 1, 2]
    assert batch["labels"][1, :2].tolist() == [-100, -100]


def test_runtime_proves_reset_positions_activate_block_diagonal_attention():
    proof = model_code.packed_attention_runtime_guard(
        SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    )
    assert proof["status"] == "PASS"
    assert proof["probe_segment_ids"] == [0, 0, 0, 1, 1, 2]


def test_forgetting_gate_checks_plan_and_apply_for_every_operation():
    def atomic(plan_overall, sh1_plan):
        plan_by = {op: 1.0 for op in core.OPS}; plan_by["SH1"] = sh1_plan
        return {"plan": {"overall": plan_overall, "by_operation": plan_by},
                "apply": {"overall": 1.0, "by_operation": {op: 1.0 for op in core.OPS}}}

    result = evaluation.aligned_forgetting(atomic(1.0, 1.0), atomic(0.99, 0.90))
    assert result["drops"]["plan_SH1"] == pytest.approx(0.10)
    assert result["passes_le_002"] is False


def test_ranking_metrics_are_exact_and_normalized():
    ranked = [
        {"rank": 1, "score": -0.1, "correct": False},
        {"rank": 2, "score": -0.2, "correct": True},
        {"rank": 3, "score": -1.0, "correct": True},
    ]
    metrics = evaluation.ranking_metrics(ranked)
    assert metrics["hit@1"] == 0.0 and metrics["hit@8"] == 1.0
    assert metrics["best_rank"] == 2 and metrics["mrr"] == 0.5
    assert 0.0 < metrics["correct_mass"] < 1.0
    assert metrics["log_gap"] == pytest.approx(-0.1)


class Batch(dict):
    __getattr__ = dict.__getitem__
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=None):
        lengths = [max(2, len(text.split())) for text in texts]; width = max(lengths)
        ids = []; masks = []
        for length in lengths:
            ids.append([0] * (width - length) + [7] * length)
            masks.append([0] * (width - length) + [1] * length)
        return Batch({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)})


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()))
    def forward(self, input_ids, position_ids=None, **kwargs):
        logits = torch.zeros((*input_ids.shape, 16), device=input_ids.device)
        logits[..., 1:6] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=input_ids.device)
        return SimpleNamespace(logits=logits)


def test_exact_score_task_enumerates_25_and_batching_is_equivalent():
    start = [1, 2, 3]; program = ("SH1", "SC2"); target = list(core.trajectory(start, program, 5)[-1])
    row = {"schema": core.TASK_SCHEMA, "task_id": "fixture", "task_fingerprint": core.canonical_task_fingerprint(5, start, target, 2),
           "split": "dev_a", "family": "A", "p": 5, "depth": 2, "degree": 2,
           "start": start, "target": target, "witness": list(program), "states": [list(s) for s in core.trajectory(start, program, 5)],
           "solutions": []}
    token_ids = {op: index + 1 for index, op in enumerate(core.OPS)}
    metric_one, ranking_one = evaluation.score_task(FakeModel(), FakeTokenizer(), token_ids, row, batch_size=1)
    metric_many, ranking_many = evaluation.score_task(FakeModel(), FakeTokenizer(), token_ids, row, batch_size=32)
    assert len(ranking_one["ranking"]) == 25
    assert ranking_one["ranking"] == ranking_many["ranking"]
    assert metric_one == metric_many
    assert [candidate["rank"] for candidate in ranking_one["ranking"]] == list(range(1, 26))


def test_dev_cycle_is_allowed_only_when_delta_is_the_sole_failed_gate():
    attempt = {"checks": {"atomic_pass": False, "registered_atomic_pass": False,
                           "exploratory_source_authorized": True,
                           "frozen_source_metrics_match": True,
                           "base_hit32_range": True, "forgetting_le_002": True,
                           "leakage_zero": True, "equal_budget": True, "correct_mass_growth": True,
                           "delta_hit32_ge_008": False}}
    assert stage4._cycle_eligible(attempt)
    attempt["checks"]["correct_mass_growth"] = False
    assert not stage4._cycle_eligible(attempt)
    attempt["checks"]["correct_mass_growth"] = True
    attempt["checks"]["delta_hit32_ge_008"] = True
    assert not stage4._cycle_eligible(attempt)


def test_attempt_identifier_is_deterministic():
    resolved = stage4._resolved_config(seed=0, lr=1e-4, epochs=2, replay=0.2)
    assert stage4._attempt_id(resolved) == stage4._attempt_id(dict(resolved))
    assert "seed0" in stage4._attempt_id(resolved)


def test_pid_alive_is_windows_safe_using_only_our_processes(monkeypatch):
    """Never probe or signal an unrelated PID; the child is owned by this test."""
    assert stage4.pid_alive(None) is False
    assert stage4.pid_alive(0) is False
    assert stage4.pid_alive(-1) is False
    if os.name == "nt":
        def destructive_kill_must_not_run(*_args, **_kwargs):
            raise AssertionError("Windows pid_alive must not call os.kill")
        monkeypatch.setattr(stage4.os, "kill", destructive_kill_must_not_run)
    assert stage4.pid_alive(os.getpid()) is True

    child = subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait(60)"])
    try:
        assert stage4.pid_alive(child.pid) is True
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert stage4.pid_alive(child.pid) is False


@pytest.mark.parametrize("access_exists,expected_count", [(False, 0), (True, 1)])
def test_frozen_final_recovers_missing_generation_receipt(tmp_path: Path, monkeypatch, access_exists: bool, expected_count: int):
    monkeypatch.setitem(stage4.PROTOCOL, "study_class", "REGISTERED_TEST")
    root = tmp_path / "stage4"
    data = root / "data"; runs = root / "runs"; manifests = root / "manifests"
    adapters = root / "adapters"
    for name, path in (("ROOT", root), ("DATA", data), ("RUNS", runs), ("MANIFESTS", manifests),
                       ("ADAPTERS", adapters), ("RANKINGS", root / "rankings"), ("REPORTS", root / "reports")):
        monkeypatch.setattr(stage4, name, path)

    attempt_id = "crash-recovery"
    attempt_path = runs / "pilot" / attempt_id / "attempt.json"
    stage4.dump_json(attempt_path, {"status": "DONE", "attempt_id": attempt_id})
    adapter_paths = {}
    for branch in ("atomic_control", "composition_distill"):
        path = adapters / branch; path.mkdir(parents=True)
        (path / "adapter.safetensors").write_bytes(branch.encode("ascii"))
        adapter_paths[branch] = str(path)
    final_path = data / "final_a.jsonl"
    stage4.write_jsonl(final_path, [{"task_id": "frozen"}])
    frozen = {
        "schema": "stage4.composition.v5.config-freeze.v1", "status": "FROZEN",
        "test_files": {"final_a.jsonl": {"sha256": stage4.sha256(final_path)}},
    }
    stage4.dump_json(root / "CONFIG_FROZEN.json", frozen)
    if access_exists:
        stage4.dump_json(manifests / "final_evaluation_access.json", {
            "status": "STARTED", "config_frozen_sha256": stage4.sha256(root / "CONFIG_FROZEN.json"),
            "test_files": frozen["test_files"],
        })
    selected = {
        "pass": True, "attempt_id": attempt_id, "resolved_config": {}, "adapters": adapter_paths,
    }

    assert not (manifests / "final_generation_receipt.json").exists()
    returned = stage4.freeze_and_generate_final(selected)
    receipt = stage4.load_json(manifests / "final_generation_receipt.json")
    assert returned == frozen
    assert receipt["status"] == "DONE"
    assert receipt["evaluation_count"] == expected_count
    assert receipt["recovered_after_freeze"] is True
    assert receipt["config_frozen_sha256"] == stage4.sha256(root / "CONFIG_FROZEN.json")


def test_frozen_final_recovery_rejects_unbound_access_marker(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(stage4.PROTOCOL, "study_class", "REGISTERED_TEST")
    root = tmp_path / "stage4"; data = root / "data"; runs = root / "runs"; manifests = root / "manifests"
    adapters = root / "adapters"
    for name, path in (("ROOT", root), ("DATA", data), ("RUNS", runs), ("MANIFESTS", manifests),
                       ("ADAPTERS", adapters), ("RANKINGS", root / "rankings"), ("REPORTS", root / "reports")):
        monkeypatch.setattr(stage4, name, path)
    attempt_id = "bad-access"; attempt_path = runs / "pilot" / attempt_id / "attempt.json"
    stage4.dump_json(attempt_path, {"status": "DONE", "attempt_id": attempt_id})
    adapter_paths = {}
    for branch in ("atomic_control", "composition_distill"):
        path = adapters / branch; path.mkdir(parents=True); (path / "adapter.safetensors").write_bytes(branch.encode())
        adapter_paths[branch] = str(path)
    final_path = data / "final_a.jsonl"; stage4.write_jsonl(final_path, [{"task_id": "frozen"}])
    frozen = {"status": "FROZEN", "test_files": {"final_a.jsonl": {"sha256": stage4.sha256(final_path)}}}
    stage4.dump_json(root / "CONFIG_FROZEN.json", frozen)
    stage4.dump_json(manifests / "final_evaluation_access.json", {
        "status": "STARTED", "config_frozen_sha256": "0" * 64, "test_files": frozen["test_files"],
    })
    selected = {"pass": True, "attempt_id": attempt_id, "resolved_config": {}, "adapters": adapter_paths}
    with pytest.raises(RuntimeError, match="not bound"):
        stage4.freeze_and_generate_final(selected)
    assert not (manifests / "final_generation_receipt.json").exists()


def test_exploratory_protocol_never_creates_final_data():
    selected = {"pass": True, "attempt_id": "must-not-freeze", "resolved_config": {}, "adapters": {}}
    assert stage4.exploratory_mode()
    with pytest.raises(RuntimeError, match="exploratory pilot cannot create"):
        stage4.freeze_and_generate_final(selected)


def _final_access_fixture(tmp_path: Path, monkeypatch, *, owner_pid: int, resume_count: int = 0):
    root = tmp_path / "stage4"; manifests = root / "manifests"
    monkeypatch.setattr(stage4, "ROOT", root)
    monkeypatch.setattr(stage4, "MANIFESTS", manifests)
    frozen = {"status": "FROZEN", "test_files": {"final_a.jsonl": {"sha256": "A" * 64}}}
    stage4.dump_json(root / "CONFIG_FROZEN.json", frozen)
    monkeypatch.setattr(stage4, "require_frozen", lambda: frozen)
    frozen_sha = stage4.sha256(root / "CONFIG_FROZEN.json")
    stage4.dump_json(manifests / "final_generation_receipt.json", {
        "status": "DONE", "config_frozen_sha256": frozen_sha,
        "test_files": frozen["test_files"], "evaluation_count": 1,
    })
    access = {
        "schema": "stage4.composition.v5.final-access.v1", "status": "STARTED",
        "config_frozen_sha256": frozen_sha, "test_files": frozen["test_files"],
        "series": "seeds_0_to_5", "owner_pid": owner_pid, "resume_count": resume_count,
    }
    stage4.dump_json(manifests / "final_evaluation_access.json", access)
    return manifests, access


def test_final_access_rejects_concurrent_live_owner(tmp_path: Path, monkeypatch):
    fake_live_owner = os.getpid() + 1000000
    manifests, original = _final_access_fixture(tmp_path, monkeypatch, owner_pid=fake_live_owner)
    observed = []
    monkeypatch.setattr(stage4, "pid_alive", lambda pid: observed.append(pid) or True)
    with pytest.raises(RuntimeError, match="already active"):
        stage4.begin_final_evaluation()
    assert observed == [fake_live_owner]
    assert stage4.load_json(manifests / "final_evaluation_access.json") == original
    assert stage4.load_json(manifests / "final_generation_receipt.json")["evaluation_count"] == 1


def test_final_access_takes_over_stale_owner_once(tmp_path: Path, monkeypatch):
    stale_owner = os.getpid() + 1000000
    manifests, _ = _final_access_fixture(tmp_path, monkeypatch, owner_pid=stale_owner, resume_count=2)
    observed = []
    monkeypatch.setattr(stage4, "pid_alive", lambda pid: observed.append(pid) or False)
    try:
        resumed = stage4.begin_final_evaluation()
        assert observed == [stale_owner]
        assert resumed["owner_pid"] == os.getpid()
        assert resumed["resume_count"] == 3
        persisted = stage4.load_json(manifests / "final_evaluation_access.json")
        assert persisted["owner_pid"] == os.getpid()
        assert persisted["resume_count"] == 3
        assert "resumed_at_unix" in persisted
        assert stage4.load_json(manifests / "final_generation_receipt.json")["evaluation_count"] == 1
    finally:
        stage4.release_final_lease()


def test_atomic_base_confirm_branch_has_explicit_zero_update_receipt(tmp_path: Path, monkeypatch):
    root = tmp_path / "stage4"; runs = root / "runs"; export = root / "atomic_export"
    export.mkdir(parents=True); (export / "model.safetensors").write_bytes(b"frozen-base")
    stage4.dump_json(root / "CONFIG_FROZEN.json", {"status": "FROZEN"})
    monkeypatch.setattr(stage4, "ROOT", root)
    monkeypatch.setattr(stage4, "RUNS", runs)
    monkeypatch.setattr(stage4, "ATOMIC_EXPORT", export)
    frozen = {"resolved_config": stage4._resolved_config(seed=0, lr=1e-4, epochs=2, replay=0.2)}
    receipt = stage4._atomic_base_zero_update_run(3, frozen)
    assert receipt["status"] == "DONE" and receipt["branch"] == "atomic_base"
    assert receipt["optimizer_steps"] == receipt["loss_bearing_target_tokens_seen"] == receipt["epochs"] == 0
    assert receipt["adapter"] is None
    assert stage4._atomic_base_zero_update_run(3, frozen) == receipt


def _ranking_cache_fixture(tmp_path: Path):
    branch = "cached_branch"
    binding = {"schema": "test.eval-binding.v1", "seed": 7, "dataset_sha256": "D" * 64}
    start = [1, 2, 3]
    witness = ("SH1", "SC2")
    target = list(core.trajectory(start, witness, 5)[-1])
    task = {
        "schema": core.TASK_SCHEMA,
        "task_id": "cached-ranking-task",
        "task_fingerprint": core.canonical_task_fingerprint(5, start, target, 2),
        "split": "dev_a",
        "family": "A",
        "p": 5,
        "depth": 2,
        "degree": 2,
        "start": start,
        "target": target,
        "witness": list(witness),
        "states": [list(state) for state in core.trajectory(start, witness, 5)],
        "solutions": [],
    }
    candidates = []
    for index, program in enumerate(core.enumerate_programs(task["depth"])):
        candidates.append({
            "program": list(program),
            "score": -float(index),
            "correct": core.trajectory(start, program, task["p"])[-1] == tuple(target),
        })
    candidates.sort(key=lambda item: (-item["score"], tuple(item["program"])))
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
    raw = {
        "schema": "stage4.composition.v5.ranking.v1",
        "task_id": task["task_id"],
        "task_fingerprint": task["task_fingerprint"],
        "split": task["split"],
        "p": task["p"],
        "depth": task["depth"],
        "start": task["start"],
        "target": task["target"],
        "branch": branch,
        "binding": binding,
        "seed": binding["seed"],
        "ranking": candidates,
    }
    metric = {
        "schema": "stage4.composition.v5.task-metrics.v1",
        "task_id": task["task_id"],
        "task_fingerprint": task["task_fingerprint"],
        "split": task["split"],
        "p": task["p"],
        "depth": task["depth"],
        **evaluation.ranking_metrics(candidates),
        "branch": branch,
        "binding": binding,
        "seed": binding["seed"],
    }
    ranking_path = tmp_path / "rankings" / branch / "part-00000.jsonl.gz"
    metric_path = tmp_path / "metrics" / branch / "part-00000.jsonl"
    receipt_path = tmp_path / "metrics" / branch / "part-00000.receipt.json"
    evaluation.write_jsonl(ranking_path, [raw], gzip_output=True)
    evaluation.write_jsonl(metric_path, [metric])
    receipt = {
        "schema": "stage4.distill.v5.eval-shard-receipt.v1",
        "status": "DONE",
        "branch": branch,
        "binding": binding,
        "task_ids": [task["task_id"]],
        "ranking_sha256": evaluation.sha256(ranking_path),
        "metrics_sha256": evaluation.sha256(metric_path),
        "ranking_rows": 1,
        "metrics_rows": 1,
        "accounting": {"model_forward_passes": 17, "prefix_sequences": 41, "input_tokens": 313},
        "wall_seconds": 1.25,
    }
    evaluation.dump_json(receipt_path, receipt)
    return task, branch, binding, raw, metric, ranking_path, metric_path, receipt_path


def test_cached_ranking_rederives_every_metric_but_preserves_receipt_accounting(tmp_path: Path):
    task, branch, binding, _raw, expected, *_paths = _ranking_cache_fixture(tmp_path)
    summary, metrics = evaluation.evaluate_ranking(
        None, None, None, [task], tmp_path, branch, binding, shard_size=25,
    )
    assert metrics == [expected]
    assert summary["accounting"] == {
        "model_forward_passes": 17,
        "prefix_sequences": 41,
        "input_tokens": 313,
    }


def test_cached_ranking_rejects_receipted_scientific_scalar_not_derived_from_raw(tmp_path: Path):
    task, branch, binding, _raw, metric, _ranking_path, metric_path, receipt_path = _ranking_cache_fixture(tmp_path)
    metric["correct_mass"] = 0.0
    evaluation.write_jsonl(metric_path, [metric])
    receipt = stage4.load_json(receipt_path)
    receipt["metrics_sha256"] = evaluation.sha256(metric_path)
    evaluation.dump_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="metrics differ from independently derived raw rankings"):
        evaluation.evaluate_ranking(None, None, None, [task], tmp_path, branch, binding)


@pytest.mark.parametrize("fault", ["task_id", "binding", "enumeration", "duplicate", "order", "correct"])
def test_cached_ranking_rejects_invalid_raw_full_ranking(tmp_path: Path, fault: str):
    task, branch, binding, raw, _metric, ranking_path, _metric_path, receipt_path = _ranking_cache_fixture(tmp_path)
    corrupted = copy.deepcopy(raw)
    if fault == "task_id":
        corrupted["task_id"] = "different-task"
    elif fault == "binding":
        corrupted["binding"]["seed"] += 1
    elif fault == "enumeration":
        corrupted["ranking"].pop()
    elif fault == "duplicate":
        corrupted["ranking"][-1]["program"] = list(corrupted["ranking"][0]["program"])
    elif fault == "order":
        corrupted["ranking"][0], corrupted["ranking"][1] = corrupted["ranking"][1], corrupted["ranking"][0]
        corrupted["ranking"][0]["rank"] = 1
        corrupted["ranking"][1]["rank"] = 2
    elif fault == "correct":
        corrupted["ranking"][0]["correct"] = not corrupted["ranking"][0]["correct"]
    evaluation.write_jsonl(ranking_path, [corrupted], gzip_output=True)
    receipt = stage4.load_json(receipt_path)
    receipt["ranking_sha256"] = evaluation.sha256(ranking_path)
    evaluation.dump_json(receipt_path, receipt)
    with pytest.raises(RuntimeError):
        evaluation.evaluate_ranking(None, None, None, [task], tmp_path, branch, binding)


def _atomic_cache_fixture(tmp_path: Path):
    branch = "cached_atomic"
    binding = {"schema": "test.atomic-binding.v1", "seed": 3, "dataset_sha256": "A" * 64}
    rows = []
    raw_rows = []
    for index, op in enumerate(core.OPS):
        start = [index + 1, 2, 3]
        row = {"task_id": f"atomic-{op}", "operation": op, "start": start, "p": 7}
        target = core.format_state(core.trajectory(start, [op], row["p"])[-1])
        prediction = core.OPS[0] if index == 1 else op
        completion = target + " trailing" if index == 2 else target
        rows.append(row)
        raw_rows.append({
            "schema": "stage4.distill.v5.atomic-generation.v1",
            "task_id": row["task_id"],
            "operation": op,
            "target": target,
            "raw_completion": completion,
            "correct": completion.strip() == target,
            "match_mode": "stripped_exact_equality",
            "plan_prediction": prediction,
            "plan_correct": prediction == op,
            "branch": branch,
            "binding": binding,
            "seed": binding["seed"],
        })
    plan = {"overall": 0.8, "by_operation": {op: float(index != 1) for index, op in enumerate(core.OPS)},
            "accounting": {"model_forward_passes": 11, "prompt_sequences": 5, "input_tokens": 101}}
    apply = {"overall": 0.8, "by_operation": {op: float(index != 2) for index, op in enumerate(core.OPS)},
             "match_mode": "stripped_exact_equality",
             "accounting": {"generation_batches": 7, "generation_decode_steps": 19,
                            "prompt_sequences": 5, "prompt_tokens": 97, "decoded_completion_tokens": 29}}
    generation_path = tmp_path / "atomic" / f"{branch}.generations.jsonl"
    result_path = tmp_path / "atomic" / f"{branch}.json"
    evaluation.write_jsonl(generation_path, raw_rows)
    result = {
        "schema": "stage4.composition.v5.atomic-eval.v1",
        "status": "DONE",
        "branch": branch,
        "binding": binding,
        "plan": plan,
        "apply": apply,
        "wall_seconds": 2.5,
        "tasks": len(rows),
        "generation_rows": len(raw_rows),
        "generation_sha256": evaluation.sha256(generation_path),
    }
    evaluation.dump_json(result_path, result)
    return rows, branch, binding, raw_rows, result, generation_path, result_path


def test_cached_atomic_rederives_plan_and_apply_but_preserves_receipt_accounting(tmp_path: Path):
    rows, branch, binding, _raw_rows, result, *_paths = _atomic_cache_fixture(tmp_path)
    observed = evaluation.evaluate_atomic(None, None, None, rows, tmp_path, branch, binding)
    assert observed["plan"]["overall"] == 0.8
    assert observed["apply"]["overall"] == 0.8
    assert observed["plan"]["accounting"] == result["plan"]["accounting"]
    assert observed["apply"]["accounting"] == result["apply"]["accounting"]


def test_cached_atomic_rejects_receipted_scalar_not_derived_from_raw(tmp_path: Path):
    rows, branch, binding, _raw_rows, result, _generation_path, result_path = _atomic_cache_fixture(tmp_path)
    result["plan"]["overall"] = 1.0
    evaluation.dump_json(result_path, result)
    with pytest.raises(RuntimeError, match="atomic scalars differ from raw per-task evidence"):
        evaluation.evaluate_atomic(None, None, None, rows, tmp_path, branch, binding)


@pytest.mark.parametrize("fault", ["count", "task_id", "binding", "plan", "apply"])
def test_cached_atomic_rejects_invalid_raw_task_evidence(tmp_path: Path, fault: str):
    rows, branch, binding, raw_rows, _result, generation_path, result_path = _atomic_cache_fixture(tmp_path)
    corrupted = copy.deepcopy(raw_rows)
    if fault == "count":
        corrupted.pop()
    elif fault == "task_id":
        corrupted[0]["task_id"] = "different-task"
    elif fault == "binding":
        corrupted[0]["binding"]["seed"] += 1
    elif fault == "plan":
        corrupted[0]["plan_prediction"] = core.OPS[1]
    elif fault == "apply":
        corrupted[0]["raw_completion"] += " altered"
    evaluation.write_jsonl(generation_path, corrupted)
    result = stage4.load_json(result_path)
    result["generation_sha256"] = evaluation.sha256(generation_path)
    evaluation.dump_json(result_path, result)
    with pytest.raises(RuntimeError):
        evaluation.evaluate_atomic(None, None, None, rows, tmp_path, branch, binding)


@pytest.mark.parametrize("field,value", [
    ("binding", {"schema": "test.atomic-binding.v1", "seed": 99}),
    ("generation_sha256", "0" * 64),
    ("generation_rows", 4),
    ("tasks", 4),
])
def test_cached_atomic_rejects_invalid_result_receipt(tmp_path: Path, field: str, value):
    rows, branch, binding, _raw_rows, result, _generation_path, result_path = _atomic_cache_fixture(tmp_path)
    result[field] = value
    evaluation.dump_json(result_path, result)
    with pytest.raises(RuntimeError, match="mismatched cached atomic evaluation"):
        evaluation.evaluate_atomic(None, None, None, rows, tmp_path, branch, binding)


def test_analysis_rows_rejects_seed_or_binding_identity_mismatch():
    row = {"task_id": "t", "seed": 0, "split": "final_a", "branch": "seed0_atomic_control_final_a",
           "binding": {"seed": 0, "branch": "atomic_control", "split": "final_a"}}
    seed_results = {0: {"atomic_control": {"metrics": {"final_a": [row]}}}}
    assert stage4._analysis_rows(seed_results, "atomic_control", ["final_a"])[0]["seed"] == 0
    bad = {0: {"atomic_control": {"metrics": {"final_a": [{**row, "seed": 1}]}}}}
    with pytest.raises(RuntimeError, match="identity/binding mismatch"):
        stage4._analysis_rows(bad, "atomic_control", ["final_a"])


def test_complete_partial_adapter_is_promoted_without_retraining(tmp_path: Path, monkeypatch):
    root = tmp_path / "stage4"; export = root / "atomic_export"; output = root / "adapters/branch"
    export.mkdir(parents=True); (export / "model.safetensors").write_bytes(b"base")
    monkeypatch.setattr(model_code, "ROOT", root)
    runtime = {"runtime_device": "cuda", "runtime_dtype": "torch.bfloat16",
               "cuda_device_name": "fixture", "torch_version": "fixture", "cuda_runtime": "fixture"}
    monkeypatch.setattr(model_code, "require_bf16_cuda_training", lambda _resolved: runtime)
    resolved = stage4._resolved_config(seed=0, lr=1e-4, epochs=2, replay=0.2)
    encoded_rows = [encoded(2, 5)]
    resolved_sha = model_code.hashlib.sha256(model_code.json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    expected = {"branch": "atomic_control", "seed": 0, "lr": resolved["lr"], "epochs": 2,
                "effective_batch": resolved["effective_batch"], "records": 1, "input_sha256": "INPUT",
                "atomic_export_sha256": model_code._export_payload_sha(export),
                "resolved_config_sha256": resolved_sha, "loss_bearing_target_tokens_per_epoch": 2,
                "loss_bearing_target_tokens_expected": 4, "input_tokens_per_epoch": 5,
                "prompt_tokens_per_epoch": 3, **runtime}
    partial = output.with_name(output.name + ".partial"); partial.mkdir(parents=True)
    (partial / "adapter_config.json").write_text("{}", encoding="utf-8")
    (partial / "adapter_model.safetensors").write_bytes(b"adapter")
    (partial / "tokenizer.json").write_text("{}", encoding="utf-8")
    model_code.dump_json(partial / "training_receipt.json", {"schema": "stage4.distill.v5.training-receipt.v1",
                                                               "status": "DONE", **expected})
    receipt = model_code.train_branch(export, output, encoded_rows, resolved, "atomic_control", "INPUT")
    assert receipt["status"] == "DONE" and output.is_dir() and not partial.exists()
    status = model_code.json.loads(model_code._training_status_path(output).read_text(encoding="utf-8"))
    assert status["status"] == "DONE" and status["recovered_from_complete_payload"] is True


def test_completed_confirmation_is_bound_to_all_raw_evidence(tmp_path: Path, monkeypatch):
    root = tmp_path / "stage4"; runs = root / "runs"; adapters = root / "adapters"
    manifests = root / "manifests"; reports = root / "reports"; export = root / "atomic_export"
    for name, path in (("ROOT", root), ("RUNS", runs), ("ADAPTERS", adapters), ("MANIFESTS", manifests),
                       ("REPORTS", reports), ("ATOMIC_EXPORT", export)):
        monkeypatch.setattr(stage4, name, path)
    selected_attempt = "selected"; control = adapters / "pilot/control"; distill = adapters / "pilot/distill"
    selection = {"selected_attempt": selected_attempt,
                 "adapters": {"atomic_control": {"path": str(control)}, "composition_distill": {"path": str(distill)}}}
    stage4.dump_json(root / "CONFIG_SELECTION_FROZEN.json", selection)
    stage4.dump_json(root / "CONFIG_FROZEN.json", {"status": "FROZEN"})
    for path in (runs / f"pilot/{selected_attempt}", adapters / "confirm", export, control, distill):
        path.mkdir(parents=True, exist_ok=True); (path / "payload.bin").write_bytes(path.name.encode())
    for seed in stage4.PROTOCOL["future_confirm_template_non_executable"]["seeds"]:
        stage4.dump_json(runs / f"confirm/seed{seed}/run.json", {"status": "DONE", "seed": seed})
    stage4.dump_json(runs / "PREPARE_DONE.json", {"status": "DONE"})
    stage4.dump_json(runs / "PILOT_DECISION.json", {"status": "PASS"})
    stage4.dump_json(reports / "confirmatory_statistics.json", {"positive_A": False, "positive_B": False})
    stage4.dump_json(manifests / "preterminal_ranking_validation.json", {"status": "PASS"})
    for name in ("preregistration.json", "atomic_reference.json", "atomic_reference_audit.json", "atomic_import.json",
                 "pilot_data_manifest.json", "pilot_split_audit.json", "pilot_generation_audit.json", "discover_audit.json",
                 "final_data_manifest.json", "final_split_audit.json", "final_generation_receipt.json"):
        stage4.dump_json(manifests / name, {"status": "DONE", "name": name})
    evidence = stage4._confirmation_evidence_payload()
    stage4.dump_json(manifests / "confirm_evidence_manifest.json", evidence)
    done = {"status": "DONE", "positive_A": False, "positive_B": False,
            "scientific_result": "NO_CONFIRMATORY_EVIDENCE_A",
            "statistics_sha256": stage4.sha256(reports / "confirmatory_statistics.json"),
            "config_frozen_sha256": stage4.sha256(root / "CONFIG_FROZEN.json"),
            "evidence_manifest_sha256": stage4.sha256(manifests / "confirm_evidence_manifest.json"),
            "raw_validation_sha256": stage4.sha256(manifests / "preterminal_ranking_validation.json")}
    done_path = runs / "STAGE4_DONE.json"; stage4.dump_json(done_path, done)
    stage4.dump_json(manifests / "final_evaluation_access.json", {
        "status": "STARTED", "config_frozen_sha256": stage4.sha256(root / "CONFIG_FROZEN.json")})
    assert stage4._validate_completed_confirmation(done_path) == done
    (runs / "confirm/seed1/run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not DONE/bound|raw evidence changed"):
        stage4._validate_completed_confirmation(done_path)

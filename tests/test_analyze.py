import json
import tempfile
from pathlib import Path

import pytest

from iclr.analyze import analyze, historical_summary, load_manifest


def test_paired_curves_and_identity():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        entries = []
        for seed in range(3):
            for arm, ranks in (("atomic_control", [40, 2]), ("composition", [1, 3])):
                rows = [dict(task_id=f"task{i}", task_fingerprint=f"fingerprint{i}", split="final_a", p=5,
                             depth=3, best_rank=rank, correct_mass=0.2, correct_count=1,
                             degree=2, sh1_any=False, sh1_required=False)
                        for i, rank in enumerate(ranks)]
                name = f"{seed}_{arm}.jsonl"
                (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
                entries.append(dict(seed=seed, arm=arm, metrics=name, scorer_id="full_vocab_op_tokens_v1",
                                    tokenizer_hash="test_tokenizer", prompt_version="stage4_plan_v1"))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(entries))
        curves = analyze(manifest, root / "output")
        assert len(curves) == 125
        for arm in ("atomic_control", "composition"):
            assert [r[arm] for r in curves] == sorted(r[arm] for r in curves)
            assert curves[-1][arm] == 1
        assert curves[-1]["delta"] == 0
        assert curves[31]["delta"] == 0.5
        assert curves[31]["sign_flip_p"] == 0.25
        assert sum(r["atomic_control"] for r in curves) / 125 == (126 - 21) / 125
        assert (root / "output" / "paired_task_deltas.csv").is_file()
        assert "composition_rank" in (root / "output" / "paired_task_deltas.csv").read_text().splitlines()[0]
        assert "composition_only" in (root / "output" / "table9_counts.csv").read_text().splitlines()[0]
        path = root / entries[-1]["metrics"]
        original = path.read_text()
        path.write_text(original.replace("fingerprint0", "different"))
        try:
            load_manifest(manifest)
        except ValueError as error:
            assert "paired task" in str(error)
        else:
            raise AssertionError("unpaired task fingerprint accepted")
        path.write_text(original)
        entries[-1]["scorer_id"] = "legal_action_softmax_v1"
        manifest.write_text(json.dumps(entries))
        try:
            load_manifest(manifest)
        except ValueError as error:
            assert "provenance" in str(error)
        else:
            raise AssertionError("incompatible scoring conventions accepted")


def test_named_contrasts_and_input_provenance(tmp_path):
    arms = ("program_only", "program_trace")
    entries = []
    for arm, rank in zip(arms, (40, 1)):
        row = dict(task_id="task", task_fingerprint="fingerprint", split="dev_A", p=5,
                   depth=3, best_rank=rank, correct_mass=.2, base_hash="same-model",
                   data_hash="same-data", model_hash=arm, training_seed=7)
        (tmp_path / f"{arm}.jsonl").write_text(json.dumps(row) + "\n")
        entries.append(dict(seed=7, arm=arm, metrics=f"{arm}.jsonl",
                            scorer_id="full_vocab_op_tokens_v1", tokenizer_hash="tokenizer",
                            prompt_version="stage4_plan_v1"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    curves = analyze(manifest, tmp_path / "output", plots=True, arms=arms)
    assert curves[31]["program_only"] == 0
    assert curves[31]["program_trace"] == 1
    assert curves[31]["delta"] == 1
    assert curves[-1]["delta"] == 0
    assert (tmp_path / "output" / "hitk_0_depth3.pdf").is_file()
    assert "treatment_rank" in (tmp_path / "output" / "paired_task_deltas.csv").read_text().splitlines()[0]
    assert "treatment_only" in (tmp_path / "output" / "table9_counts.csv").read_text().splitlines()[0]
    assert json.loads((tmp_path / "output" / "analysis.json").read_text())["arms"] == {
        "control": "program_only", "treatment": "program_trace"}
    metric = tmp_path / "program_trace.jsonl"
    original = metric.read_text()
    for key, value in (("base_hash", "other-model"), ("data_hash", "other-data"), ("training_seed", 8)):
        row = {**json.loads(original), key: value}
        metric.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="provenance|seed mismatch"):
            load_manifest(manifest, arms)
    metric.write_text(original)


def test_all_arms_cannot_silently_drop_the_same_task(tmp_path):
    from composition_core import trajectory
    from iclr.data import correct_programs

    tasks = [dict(task_id=f"task{i}", task_fingerprint=f"fingerprint{i}", split="dev_A",
                  depth=3, p=5, start=[1, 2, i + 1],
                  target=list(trajectory([1, 2, i + 1], ("SH1", "SH1", "SH1"), 5)[-1]))
             for i in range(2)]
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(task) + "\n" for task in tasks))
    entries = []
    for arm in ("atomic_control", "composition"):
        row = {**tasks[0], "best_rank": 1, "correct_mass": .2,
               "correct_count": len(correct_programs(tasks[0]))}
        (tmp_path / f"{arm}.jsonl").write_text(json.dumps(row) + "\n")
        entries.append(dict(seed=0, arm=arm, metrics=f"{arm}.jsonl", tasks="tasks.jsonl",
                            scorer_id="full_vocab_op_tokens_v1", tokenizer_hash="tokenizer",
                            prompt_version="stage4_plan_v1"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="full task file"):
        load_manifest(manifest)


def test_analysis_rejects_mixed_precision_and_mixed_checkpoints(tmp_path):
    entries = []
    for arm in ("atomic_control", "composition"):
        rows = [dict(task_id=f"task{i}", task_fingerprint=f"fingerprint{i}", split="dev_A", p=5,
                     depth=3, best_rank=1, correct_mass=.5, base_hash="base", data_hash="data",
                     adapter_hash=arm, dtype="float32") for i in range(2)]
        (tmp_path / f"{arm}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        entries.append(dict(seed=0, arm=arm, metrics=f"{arm}.jsonl", scorer_id="full_vocab_op_tokens_v1",
                            tokenizer_hash="tokenizer", prompt_version="stage4_plan_v1"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    assert load_manifest(manifest)[1]["dtype"] == "float32"
    metric = tmp_path / "composition.jsonl"
    original = metric.read_text()
    for key, value, indices in (("adapter_hash", "another-checkpoint", [1]),
                                ("dtype", "bfloat16", [0, 1]), ("dtype", None, [0, 1])):
        rows = [json.loads(line) for line in original.splitlines()]
        for index in indices:
            rows[index][key] = value
        metric.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError, match="provenance"):
            load_manifest(manifest)
    metric.write_text(original)


def test_published_table9():
    source = Path(__file__).resolve().parents[1] / "evidence" / "PRIMARY_ANALYSIS.json"
    primary = json.loads(source.read_text())["primary"]
    # Submitted PDF Table 9, p. 17, in the order both/control-only/composition-only/neither.
    expected = [(305, 130, 432, 133), (316, 141, 408, 135), (305, 149, 409, 137),
                (343, 117, 409, 131), (321, 138, 408, 133), (325, 129, 382, 164)]
    assert [tuple(r[k] for k in ("both_hit", "control_only", "distill_only", "neither"))
            for r in primary["per_seed"]] == expected
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "aggregates"
        result = historical_summary(source, output)
        assert round(100 * result["atomic_control_hit32"], 2) == 45.32
        assert round(100 * result["composition_hit32"], 2) == 72.72
        assert result["seed_statistics"]["mean_delta"] == 0.274
        assert result["exact_sign_flip"]["p_two_sided"] == 0.03125
        assert not list(output.glob("hitk*"))


if __name__ == "__main__":
    test_paired_curves_and_identity()
    test_published_table9()

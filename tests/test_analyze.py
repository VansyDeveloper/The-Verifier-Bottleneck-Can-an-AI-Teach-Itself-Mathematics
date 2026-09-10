import json
import tempfile
from pathlib import Path

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

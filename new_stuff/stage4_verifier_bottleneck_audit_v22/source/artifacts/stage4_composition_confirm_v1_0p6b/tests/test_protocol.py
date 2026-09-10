import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_schema_and_model_lock():
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "schemas/protocol.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(protocol)
    assert protocol["model"] == "Qwen/Qwen3-0.6B"
    assert protocol["model_size_lock"] == "0.6B"
    assert "1.7B" in protocol["forbidden_model_sizes"]


def test_six_fresh_confirmation_streams():
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    labels = [row["replicate_label"] for row in protocol["replicates"]]
    seeds = [row["rng_seed"] for row in protocol["replicates"]]
    assert labels == list(range(6))
    assert len(set(seeds)) == 6
    assert 0 not in seeds
    assert protocol["seed0_pilot_adapter_reuse"] is False


def test_exact_program_counts_and_composition_only_gate():
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    assert protocol["ranking"]["candidate_counts"] == {"2": 25, "3": 125, "4": 625}
    assert protocol["primary"]["endpoint"] == "final_a_depth3_hit@32"
    assert protocol["primary"]["mean_delta_min"] == 0.05
    assert protocol["non_gating"]["atomic_plan_apply_and_sh1_forgetting"] is True


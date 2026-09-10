import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import confirm


def test_external_runtime_dependencies_are_frozen():
    manifest = confirm.code_manifest()
    required_suffixes = {
        "composition_core.py", "composition_eval.py", "composition_model.py",
        "v5_train.py", "v2_atomic.py", "v3_train.py", "stage4_core.py",
        "stage4_eval.py", "stage4_model.py",
    }
    external = {key.rsplit("/", 1)[-1] for key in manifest if key.startswith("external:")}
    assert required_suffixes <= external


def test_source_audit_binds_export_pilot_and_tokenizer_policy():
    audit = confirm.source_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["tokenizer_regex_policy"] == "preserve_frozen_export_default_used_by_pilot"


def test_confirmation_runner_has_exact_matrix_constants():
    source = (ROOT / "code/confirm.py").read_text(encoding="utf-8")
    assert 'expected_shards = 480 if mode == "primary" else 2080' in source
    assert 'validate_raw("primary")' in source


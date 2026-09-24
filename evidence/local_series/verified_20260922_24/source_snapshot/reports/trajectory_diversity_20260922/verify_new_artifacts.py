"""Independent integrity pass for the completed trajectory-selection experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "artifacts/trajectory_diversity_20260922"
LEGACY = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8"
sys.path.insert(0, str(LEGACY / "code"))
import composition_model as model_lib


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def main() -> None:
    freeze_path = ROOT / "SELECTION_FROZEN.json"
    freeze_sha = sha256(freeze_path)
    adapters = []
    for seed in range(83000, 83006):
        pair_path = ROOT / "runs" / f"seed{seed}_pair.json"
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        if pair.get("status") != "DONE" or pair.get("freeze_sha256", "").upper() != freeze_sha:
            raise RuntimeError(f"seed {seed}: invalid pair receipt")
        for arm in ("random", "diverse"):
            adapter = ROOT / "adapters" / f"seed{seed}_{arm}"
            receipt = json.loads((adapter / "training_receipt.json").read_text(encoding="utf-8"))
            expected = pair["adapters"][arm]
            actual_tree = model_lib.tree_sha256(adapter).upper()
            if sha256(adapter / "training_receipt.json") != expected["receipt_sha256"].upper():
                raise RuntimeError(f"seed {seed} {arm}: training receipt SHA-256 changed")
            if actual_tree != expected["tree_sha256"].upper():
                raise RuntimeError(f"seed {seed} {arm}: adapter tree SHA-256 changed")
            if receipt.get("status") != "DONE" or receipt.get("optimizer_steps") != 30:
                raise RuntimeError(f"seed {seed} {arm}: invalid training completion")
            adapters.append({"seed": seed, "arm": arm, "tree_sha256": actual_tree,
                             "receipt_sha256": expected["receipt_sha256"].upper()})
    output = {"schema": "trajectory-diversity.integrity-audit.v1", "status": "DONE",
              "freeze_sha256": freeze_sha, "verified_adapters": len(adapters),
              "adapters": adapters}
    path = HERE / "data/new_experiment/integrity_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

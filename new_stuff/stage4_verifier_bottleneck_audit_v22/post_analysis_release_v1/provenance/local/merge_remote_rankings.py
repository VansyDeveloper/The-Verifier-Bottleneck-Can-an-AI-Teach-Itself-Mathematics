from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


REPO = Path(r"C:\MyProject\verifier_bottleneck_codex_pack")
LOCAL_ROOT = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
sys.path.insert(0, str(LOCAL_ROOT / "code"))

import confirm  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_paths(raw_relative: Path) -> tuple[Path, Path]:
    parts = list(raw_relative.parts)
    ranking_index = parts.index("rankings", 2)
    parts[ranking_index] = "metrics"
    metric_name = parts[-1]
    if not metric_name.endswith(".jsonl.gz"):
        raise RuntimeError(f"unexpected ranking filename: {raw_relative}")
    parts[-1] = metric_name[:-3]
    metric = Path(*parts)
    receipt = metric.with_name(metric.stem + ".receipt.json")
    return metric, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    remote_root = args.remote_root.resolve()
    if not remote_root.is_dir():
        raise RuntimeError(f"remote root is absent: {remote_root}")
    remote_rankings = remote_root / "rankings"
    raw_files = sorted(remote_rankings.glob("replicate*/rankings/*/part-*.jsonl.gz"))
    if not raw_files:
        raise RuntimeError("downloaded remote tree has no raw ranking shards")

    task_cache: dict[str, dict[str, dict]] = {}
    reused = 0
    imported = 0
    rows_imported = 0
    reports = []
    started = time.time()
    for remote_raw in raw_files:
        relative = remote_raw.relative_to(remote_root)
        local_raw = LOCAL_ROOT / relative
        remote_metric_relative, remote_receipt_relative = metric_paths(relative)
        remote_metric = remote_root / remote_metric_relative
        remote_receipt = remote_root / remote_receipt_relative
        if not remote_metric.is_file() or not remote_receipt.is_file():
            raise RuntimeError(f"remote shard is incomplete: {remote_raw}")
        receipt = json.loads(remote_receipt.read_text(encoding="utf-8"))
        remote_raw_sha = sha256(remote_raw)
        if receipt.get("status") != "DONE" or receipt.get("ranking_sha256") != remote_raw_sha:
            raise RuntimeError(f"remote ranking receipt mismatch: {remote_receipt}")

        raw_rows = confirm.eval_lib.read_jsonl(remote_raw)
        if len(raw_rows) != receipt.get("ranking_rows") or not raw_rows:
            raise RuntimeError(f"remote ranking row count mismatch: {remote_raw}")
        split = raw_rows[0]["split"]
        if any(row.get("split") != split for row in raw_rows):
            raise RuntimeError(f"mixed split in ranking shard: {remote_raw}")
        if split not in task_cache:
            source_rows = confirm.read_jsonl(confirm.DATA / f"{split}.jsonl")
            task_cache[split] = {row["task_id"]: row for row in source_rows}
        tasks = []
        for raw in raw_rows:
            task = task_cache[split].get(raw.get("task_id"))
            if task is None:
                raise RuntimeError(f"unknown task in remote ranking: {raw.get('task_id')}")
            tasks.append(task)

        branch = raw_rows[0]["branch"]
        binding = raw_rows[0]["binding"]
        derived = [
            confirm.eval_lib._ranking_metric_from_raw(task, raw, branch, binding)
            for task, raw in zip(tasks, raw_rows)
        ]
        if receipt.get("task_ids") != [row["task_id"] for row in tasks]:
            raise RuntimeError(f"remote receipt task order mismatch: {remote_receipt}")

        local_metric = LOCAL_ROOT / remote_metric_relative
        local_receipt = LOCAL_ROOT / remote_receipt_relative
        if local_raw.exists():
            if sha256(local_raw) != remote_raw_sha:
                raise RuntimeError(f"existing local raw shard conflicts with remote: {relative}")
            reused += 1
        else:
            local_raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_raw, local_raw)
            imported += 1
            rows_imported += len(raw_rows)

        local_metric.parent.mkdir(parents=True, exist_ok=True)
        confirm.eval_lib.write_jsonl(local_metric, derived)
        normalized_receipt = dict(receipt)
        normalized_receipt["ranking_sha256"] = sha256(local_raw)
        normalized_receipt["metrics_sha256"] = confirm.eval_lib.sha256(local_metric)
        normalized_receipt["ranking_rows"] = len(raw_rows)
        normalized_receipt["metrics_rows"] = len(derived)
        confirm.eval_lib.dump_json(local_receipt, normalized_receipt)
        checked = confirm.eval_lib._rederive_cached_ranking_shard(
            tasks, local_raw, local_metric, branch, binding
        )
        if checked != derived:
            raise RuntimeError(f"Windows normalized shard failed rederivation: {local_raw}")
        reports.append({
            "path": relative.as_posix(), "rows": len(raw_rows),
            "ranking_sha256": normalized_receipt["ranking_sha256"],
            "metrics_sha256": normalized_receipt["metrics_sha256"],
            "source_receipt_sha256": sha256(remote_receipt),
            "source_metrics_sha256": sha256(remote_metric),
            "reused_local_raw": local_raw.exists() and sha256(local_raw) == remote_raw_sha,
        })

    payload = {
        "schema": "stage4.ai01.windows-ranking-normalization.v1",
        "status": "PASS", "remote_root": str(remote_root),
        "started_at_unix": started, "finished_at_unix": time.time(),
        "raw_shards_seen": len(raw_files), "raw_shards_imported": imported,
        "raw_shards_reused": reused, "rows_imported": rows_imported,
        "reports": reports,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(args.receipt.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.receipt)
    print(json.dumps({key: value for key, value in payload.items() if key != "reports"}, sort_keys=True))


if __name__ == "__main__":
    main()

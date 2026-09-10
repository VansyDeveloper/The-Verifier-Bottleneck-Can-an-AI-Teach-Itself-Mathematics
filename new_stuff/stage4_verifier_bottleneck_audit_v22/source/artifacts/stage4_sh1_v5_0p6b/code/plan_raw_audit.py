from __future__ import annotations

import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import v5_train as v5


def write_or_verify_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if v5.read_jsonl(path) != records:
            raise RuntimeError(f"existing audit output differs from deterministic replay: {path}")
        return
    partial = path.with_name(path.name + f".{v5.os.getpid()}.{time.time_ns()}.partial")
    v5.write_jsonl(partial, records); partial.replace(path)


def derive_plan_metrics(records: list[dict]) -> dict:
    if not records:
        raise RuntimeError("PLAN raw audit records are empty")
    task_ids = [record["task_id"] for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("PLAN raw audit contains duplicate task IDs")
    by_operation = defaultdict(lambda: [0, 0]); correct = 0
    for record in records:
        operation = record["target_operation"]; ok = bool(record["correct"])
        if operation not in v5.OPS or record["predicted_operation"] not in v5.OPS:
            raise RuntimeError("PLAN raw audit contains an unknown operation")
        if ok != (record["predicted_operation"] == operation):
            raise RuntimeError("PLAN raw audit correctness flag is inconsistent")
        correct += int(ok); by_operation[operation][0] += int(ok); by_operation[operation][1] += 1
    if set(by_operation) != set(v5.OPS):
        raise RuntimeError("PLAN raw audit operation coverage is incomplete")
    return {"overall": correct / len(records),
            "by_operation": {operation: by_operation[operation][0] / by_operation[operation][1]
                             for operation in v5.OPS}}


def audit_apply_tokens(model, rows: list[dict], run_id: str) -> dict:
    device = next(model.parameters()).device
    gate_raw_path = v5.RUNS / "generations" / f"{run_id}.jsonl"
    gate_raw_rows = v5.read_jsonl(gate_raw_path); gate_raw = {row["task_id"]: row for row in gate_raw_rows}
    if len(gate_raw) != len(gate_raw_rows) or set(gate_raw) != {row["task_id"] for row in rows}:
        raise RuntimeError(f"APPLY gate raw inventory mismatch: {run_id}")
    records = []; prompt_tokens = generation_batches = generation_forward_passes = generated_token_slots = 0
    with v5.v2.torch.inference_mode():
        for at in range(0, len(rows), 8):
            chunk = rows[at:at + 8]; prompts = []
            for row in chunk:
                prompts.append(v5.apply_prompt({**row, "program": [row["witness"][0]]}))
            encoded = v5.v2.TOKENIZER(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            prompt_width = encoded.input_ids.shape[1]
            output = model.generate(**encoded, max_new_tokens=20, do_sample=False,
                                    pad_token_id=v5.v2.TOKENIZER.pad_token_id,
                                    eos_token_id=v5.v2.TOKENIZER.eos_token_id)
            suffix = output[:, prompt_width:]; texts = v5.v2.TOKENIZER.batch_decode(suffix, skip_special_tokens=True)
            prompt_tokens += int(encoded.attention_mask.sum().item()); generation_batches += 1
            generation_forward_passes += int(suffix.shape[1]); generated_token_slots += int(suffix.numel())
            for row, text, token_ids in zip(chunk, texts, suffix.tolist()):
                prior = gate_raw[row["task_id"]]
                if text != prior["raw_completion"]:
                    raise RuntimeError(f"APPLY audit replay differs from gate raw: {run_id} {row['task_id']}")
                records.append({"schema": "stage4.sh1.v5.apply-token-audit.v1", "task_id": row["task_id"],
                                "operation": row["witness"][0], "raw_completion": text,
                                "generated_token_ids": [int(value) for value in token_ids],
                                "generated_token_slots": len(token_ids), "matches_gate_raw": True})
    output_path = v5.RUNS / "audit/apply" / f"{run_id}.jsonl"; write_or_verify_jsonl(output_path, records)
    return {"raw_path": output_path.relative_to(v5.REPO).as_posix(), "raw_sha256": v5.sha256(output_path),
            "rows": len(records), "gate_raw_sha256": v5.sha256(gate_raw_path),
            "prompt_tokens": prompt_tokens, "generation_batches": generation_batches,
            "generation_forward_passes": generation_forward_passes,
            "generated_token_slots": generated_token_slots, "matches_gate_raw": True}


def audit_adapter(adapter: Path, run_id: str) -> dict:
    gate_path = v5.RUNS / f"{run_id}_gate.json"
    gate = v5.load_bound_gate(gate_path, adapter, "calibration", run_id)
    rows = v5.read_jsonl(v5.DATA / "calibration.jsonl")
    v5.configure_v2(); model, _ = v5.v2.load_continuation(adapter)
    model.float(); model.eval(); device = next(model.parameters()).device
    candidates = v5.v2.torch.tensor([v5.v2.TIDS[operation] for operation in v5.OPS], device=device)
    records = []; input_tokens = 0; forward_batches = 0; started = time.time()
    with v5.v2.torch.inference_mode():
        for at in range(0, len(rows), 16):
            chunk = rows[at:at + 16]; prompts = []
            for row in chunk:
                operation = row["witness"][0]
                end = v5.trajectory(row["start"], [operation], row["p"])[-1]
                prompts.append(v5.plan_prompt({**row, "depth": 1, "target": list(end)}))
            encoded = v5.v2.TOKENIZER(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            position_ids = encoded.attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(encoded.attention_mask == 0, 0)
            candidate_logits = model(**encoded, position_ids=position_ids).logits[:, -1][:, candidates].float().cpu()
            predictions = candidate_logits.argmax(-1).tolist(); input_tokens += int(encoded.attention_mask.sum().item()); forward_batches += 1
            for row, prediction, logits in zip(chunk, predictions, candidate_logits.tolist()):
                target = row["witness"][0]; predicted = v5.OPS[prediction]
                records.append({"schema": "stage4.sh1.v5.plan-raw-audit.v1", "task_id": row["task_id"],
                                "target_operation": target, "predicted_operation": predicted,
                                "correct": predicted == target,
                                "candidate_token_ids": {operation: v5.v2.TIDS[operation] for operation in v5.OPS},
                                "candidate_logits": {operation: float(logit) for operation, logit in zip(v5.OPS, logits)}})
    metrics = derive_plan_metrics(records)
    if metrics.keys() != gate["plan"].keys() or not math.isclose(metrics["overall"], gate["plan"]["overall"], abs_tol=1e-15):
        raise RuntimeError(f"PLAN raw audit overall mismatch: {run_id}")
    for operation in v5.OPS:
        if not math.isclose(metrics["by_operation"][operation], gate["plan"]["by_operation"][operation], abs_tol=1e-15):
            raise RuntimeError(f"PLAN raw audit operation mismatch: {run_id} {operation}")
    output = v5.RUNS / "audit/plan" / f"{run_id}.jsonl"; write_or_verify_jsonl(output, records)
    apply_audit = audit_apply_tokens(model, rows, run_id)
    result = {"run_id": run_id, "adapter": adapter.relative_to(v5.REPO).as_posix(),
              "adapter_tree_sha256": v5.tree_sha256(adapter),
              "gate_path": gate_path.relative_to(v5.REPO).as_posix(), "gate_sha256": v5.sha256(gate_path),
              "dataset_sha256": v5.verify_frozen_dataset("calibration"),
              "raw_path": output.relative_to(v5.REPO).as_posix(), "raw_sha256": v5.sha256(output), "rows": len(records),
              "metrics": metrics, "matches_registered_gate": True, "model_forward_batches": forward_batches,
              "input_tokens": input_tokens + apply_audit["prompt_tokens"],
              "plan_input_tokens": input_tokens, "plan_forward_passes": forward_batches,
              "apply_token_audit": apply_audit,
              "model_forward_passes": forward_batches + apply_audit["generation_forward_passes"],
              "generated_token_slots": apply_audit["generated_token_slots"], "wall_seconds": time.time() - started}
    del model; gc.collect()
    if v5.v2.torch.cuda.is_available(): v5.v2.torch.cuda.empty_cache()
    return result


def main() -> None:
    receipt_path = v5.MANIFESTS / "plan_raw_audit.json"
    if receipt_path.exists():
        raise RuntimeError("refusing to overwrite a completed PLAN raw audit")
    specs = [(v5.ADAPTERS / "curriculum", "curriculum_calibration")]
    if (v5.ADAPTERS / "corrective").is_dir() and (v5.RUNS / "corrective_calibration_gate.json").is_file():
        specs.append((v5.ADAPTERS / "corrective", "corrective_calibration"))
    audits = [audit_adapter(adapter, run_id) for adapter, run_id in specs]
    receipt = {"schema": "stage4.sh1.v5.plan-raw-audit-receipt.v1", "status": "DONE",
               "model": v5.CFG["model"], "audit_code_sha256": v5.sha256(Path(__file__)),
               "runs": audits, "final_dataset_accessed": False}
    v5.dump_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

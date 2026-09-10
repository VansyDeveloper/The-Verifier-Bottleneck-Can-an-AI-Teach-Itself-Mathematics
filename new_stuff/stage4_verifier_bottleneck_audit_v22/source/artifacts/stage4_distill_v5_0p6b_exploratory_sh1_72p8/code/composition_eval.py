from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from composition_core import OPS, TOKENS, apply_prompt, enumerate_programs, format_state, plan_prompt, trajectory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows, gzip_output: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    opener = gzip.open if gzip_output or path.suffix == ".gz" else open
    with opener(partial, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    partial.replace(path)


def read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def ranking_metrics(ranked):
    correct_rows = [row for row in ranked if row["correct"]]
    incorrect_rows = [row for row in ranked if not row["correct"]]
    if not correct_rows: raise RuntimeError("task has no exact correct program")
    best_rank = min(row["rank"] for row in correct_rows)
    all_scores = [row["score"] for row in ranked]; correct_scores = [row["score"] for row in correct_rows]
    metrics = {f"hit@{k}": float(best_rank <= k) for k in (1, 8, 16, 32, 64)}
    metrics.update({"correct_mass": math.exp(logsumexp(correct_scores) - logsumexp(all_scores)),
                    "best_rank": best_rank, "mrr": 1.0 / best_rank,
                    "log_gap": max(correct_scores) - max(row["score"] for row in incorrect_rows) if incorrect_rows else math.inf})
    return metrics


@torch.inference_mode()
def score_task(model, tokenizer, token_ids, row: dict, batch_size: int = 32, accounting: dict | None = None):
    device = next(model.parameters()).device; depth = int(row["depth"]); base = plan_prompt(row)
    accounting = accounting if accounting is not None else {}
    accounting.setdefault("model_forward_passes", 0); accounting.setdefault("prefix_sequences", 0)
    accounting.setdefault("input_tokens", 0)
    prefix_scores = {(): 0.0}
    for level in range(depth):
        prefixes = sorted(prefix_scores)
        next_scores = {}
        for start in range(0, len(prefixes), batch_size):
            chunk = prefixes[start:start + batch_size]
            texts = [base + (" " + " ".join(TOKENS[op] for op in prefix) if prefix else "") for prefix in chunk]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            accounting["model_forward_passes"] += 1; accounting["prefix_sequences"] += len(chunk)
            accounting["input_tokens"] += int(encoded.attention_mask.sum())
            positions = encoded.attention_mask.long().cumsum(-1) - 1
            positions.masked_fill_(encoded.attention_mask == 0, 0)
            logits = model(**encoded, position_ids=positions).logits[:, -1].float()
            log_prob = torch.log_softmax(logits, dim=-1)
            for index, prefix in enumerate(chunk):
                for op in OPS:
                    next_scores[prefix + (op,)] = prefix_scores[prefix] + float(log_prob[index, token_ids[op]])
        prefix_scores = next_scores
    programs = enumerate_programs(depth)
    candidates = []
    for program in programs:
        correct = trajectory(row["start"], program, row["p"])[-1] == tuple(row["target"])
        candidates.append({"program": list(program), "score": prefix_scores[program], "correct": bool(correct)})
    candidates.sort(key=lambda item: (-item["score"], tuple(item["program"])))
    for rank, candidate in enumerate(candidates, 1): candidate["rank"] = rank
    metrics = ranking_metrics(candidates)
    metric_row = {"schema": "stage4.composition.v5.task-metrics.v1", "task_id": row["task_id"],
                  "task_fingerprint": row["task_fingerprint"], "split": row["split"], "p": row["p"],
                  "depth": depth, **metrics}
    ranking_row = {"schema": "stage4.composition.v5.ranking.v1", "task_id": row["task_id"],
                   "task_fingerprint": row["task_fingerprint"], "split": row["split"], "p": row["p"],
                   "depth": depth, "start": row["start"], "target": row["target"], "ranking": candidates}
    return metric_row, ranking_row


def _binding_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{label} is not finite")
    return value


def _ranking_metric_from_raw(task: dict, raw: dict, branch: str, binding: dict) -> dict:
    """Validate one full raw ranking and independently derive its task metrics."""
    expected_metadata = {
        "schema": "stage4.composition.v5.ranking.v1",
        "task_id": task["task_id"],
        "task_fingerprint": task["task_fingerprint"],
        "split": task["split"],
        "p": task["p"],
        "depth": int(task["depth"]),
        "start": task["start"],
        "target": task["target"],
        "branch": branch,
    }
    for key, expected in expected_metadata.items():
        if raw.get(key) != expected:
            raise RuntimeError(f"raw ranking {task['task_id']} {key} mismatch")
    if not _binding_equal(raw.get("binding"), binding):
        raise RuntimeError(f"raw ranking {task['task_id']} binding mismatch")
    if "seed" in binding and raw.get("seed") != int(binding["seed"]):
        raise RuntimeError(f"raw ranking {task['task_id']} seed mismatch")

    depth = int(task["depth"])
    expected_programs = set(enumerate_programs(depth))
    candidates = raw.get("ranking")
    if not isinstance(candidates, list) or len(candidates) != 5 ** depth:
        raise RuntimeError(f"raw ranking {task['task_id']} does not contain exactly 5^{depth} programs")
    seen = set()
    normalized = []
    for expected_rank, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, dict):
            raise RuntimeError(f"raw ranking {task['task_id']} contains a non-object candidate")
        program_value = candidate.get("program")
        if not isinstance(program_value, list):
            raise RuntimeError(f"raw ranking {task['task_id']} candidate program is not a list")
        program = tuple(program_value)
        if program not in expected_programs or program in seen:
            raise RuntimeError(f"raw ranking {task['task_id']} has a missing, duplicate, or invalid program")
        seen.add(program)
        if isinstance(candidate.get("rank"), bool) or candidate.get("rank") != expected_rank:
            raise RuntimeError(f"raw ranking {task['task_id']} rank order is not contiguous")
        score = _number(candidate.get("score"), f"raw ranking {task['task_id']} score")
        expected_correct = trajectory(task["start"], program, task["p"])[-1] == tuple(task["target"])
        if not isinstance(candidate.get("correct"), bool) or candidate["correct"] != expected_correct:
            raise RuntimeError(f"raw ranking {task['task_id']} exact-correct flag mismatch")
        normalized.append({"program": list(program), "score": score,
                           "correct": expected_correct, "rank": expected_rank})
    if seen != expected_programs:
        raise RuntimeError(f"raw ranking {task['task_id']} program enumeration is incomplete")
    expected_order = sorted(normalized, key=lambda item: (-item["score"], tuple(item["program"])))
    if [tuple(item["program"]) for item in normalized] != [tuple(item["program"]) for item in expected_order]:
        raise RuntimeError(f"raw ranking {task['task_id']} score/tie-break order mismatch")

    metrics = ranking_metrics(normalized)
    metric_row = {"schema": "stage4.composition.v5.task-metrics.v1", "task_id": task["task_id"],
                  "task_fingerprint": task["task_fingerprint"], "split": task["split"], "p": task["p"],
                  "depth": depth, **metrics, "branch": branch, "binding": binding}
    if "seed" in binding:
        metric_row["seed"] = int(binding["seed"])
    return metric_row


def _rederive_cached_ranking_shard(shard_rows: list[dict], ranking_path: Path, metric_path: Path,
                                    branch: str, binding: dict) -> list[dict]:
    raw_rankings = read_jsonl(ranking_path)
    stored_metrics = read_jsonl(metric_path)
    task_ids = [row["task_id"] for row in shard_rows]
    if [row.get("task_id") for row in raw_rankings] != task_ids:
        raise RuntimeError(f"cached raw ranking task order mismatch: {ranking_path}")
    if [row.get("task_id") for row in stored_metrics] != task_ids:
        raise RuntimeError(f"cached metric task order mismatch: {metric_path}")
    derived = [_ranking_metric_from_raw(task, raw, branch, binding)
               for task, raw in zip(shard_rows, raw_rankings)]
    if stored_metrics != derived:
        raise RuntimeError(f"cached metrics differ from independently derived raw rankings: {metric_path}")
    return derived


def evaluate_ranking(model, tokenizer, token_ids, rows: list[dict], out_dir: Path, branch: str,
                     binding: dict, batch_size: int = 32, shard_size: int = 25):
    out_dir = Path(out_dir); ranking_dir = out_dir / "rankings" / branch; metric_dir = out_dir / "metrics" / branch
    ranking_dir.mkdir(parents=True, exist_ok=True); metric_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status" / f"{branch}.json"
    dump_json(status_path, {"schema": "stage4.distill.v5.eval-status.v1", "status": "STARTED", "branch": branch,
                            "binding": binding, "tasks": len(rows), "started_at_unix": time.time()})
    started = time.time(); all_metrics = []
    total_accounting = {"model_forward_passes": 0, "prefix_sequences": 0, "input_tokens": 0}
    total_eval_wall = 0.0
    try:
        for shard_index, offset in enumerate(range(0, len(rows), shard_size)):
            shard_rows = rows[offset:offset + shard_size]; task_ids = [row["task_id"] for row in shard_rows]
            ranking_path = ranking_dir / f"part-{shard_index:05d}.jsonl.gz"
            metric_path = metric_dir / f"part-{shard_index:05d}.jsonl"
            receipt_path = metric_dir / f"part-{shard_index:05d}.receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                valid = (receipt.get("status") == "DONE" and receipt.get("branch") == branch and
                         receipt.get("task_ids") == task_ids and
                         _binding_equal(receipt.get("binding"), binding) and ranking_path.exists() and metric_path.exists() and
                         receipt.get("ranking_sha256") == sha256(ranking_path) and receipt.get("metrics_sha256") == sha256(metric_path) and
                         receipt.get("ranking_rows") == receipt.get("metrics_rows") == len(task_ids))
                if not valid: raise RuntimeError(f"mismatched cached evaluation shard: {receipt_path}")
                shard_accounting = receipt.get("accounting")
                if not isinstance(shard_accounting, dict): raise RuntimeError(f"cached shard lacks accounting: {receipt_path}")
                for key in total_accounting: total_accounting[key] += int(shard_accounting[key])
                total_eval_wall += float(receipt["wall_seconds"])
                all_metrics.extend(_rederive_cached_ranking_shard(shard_rows, ranking_path, metric_path,
                                                                   branch, binding)); continue
            metrics = []; rankings = []; shard_accounting = {key: 0 for key in total_accounting}; shard_started = time.time()
            for row in shard_rows:
                metric, ranking = score_task(model, tokenizer, token_ids, row, batch_size, shard_accounting)
                metric["branch"] = branch; ranking["branch"] = branch; metric["binding"] = binding; ranking["binding"] = binding
                if "seed" in binding: metric["seed"] = int(binding["seed"]); ranking["seed"] = int(binding["seed"])
                derived = _ranking_metric_from_raw(row, ranking, branch, binding)
                if metric != derived: raise RuntimeError(f"fresh metrics differ from raw ranking: {row['task_id']}")
                metrics.append(derived); rankings.append(ranking)
            write_jsonl(ranking_path, rankings, gzip_output=True); write_jsonl(metric_path, metrics)
            shard_wall = time.time() - shard_started
            dump_json(receipt_path, {"schema": "stage4.distill.v5.eval-shard-receipt.v1", "status": "DONE",
                                    "branch": branch, "binding": binding, "task_ids": task_ids,
                                    "ranking_sha256": sha256(ranking_path), "metrics_sha256": sha256(metric_path),
                                    "ranking_rows": len(rankings), "metrics_rows": len(metrics),
                                    "accounting": shard_accounting, "wall_seconds": shard_wall})
            for key in total_accounting: total_accounting[key] += int(shard_accounting[key])
            total_eval_wall += shard_wall
            all_metrics.extend(metrics)
        keys = ("hit@1", "hit@8", "hit@16", "hit@32", "hit@64", "correct_mass", "best_rank", "mrr", "log_gap")
        summary = {"schema": "stage4.distill.v5.ranking-summary.v1", "branch": branch, "binding": binding,
                   "n_tasks": len(all_metrics), "wall_seconds": total_eval_wall,
                   "accounting": total_accounting,
                   "candidate_program_scores": sum(5 ** int(row["depth"]) for row in all_metrics)}
        for key in keys:
            values = [float(row[key]) for row in all_metrics if math.isfinite(float(row[key]))]
            summary[key] = sum(values) / len(values) if values else None
        dump_json(out_dir / "summaries" / f"{branch}.json", summary)
        dump_json(status_path, {"schema": "stage4.distill.v5.eval-status.v1", "status": "DONE", "branch": branch,
                                "binding": binding, "tasks": len(rows), "wall_seconds": time.time() - started})
        return summary, all_metrics
    except Exception as exc:
        dump_json(status_path, {"schema": "stage4.distill.v5.eval-status.v1", "status": "FAILED", "branch": branch,
                                "binding": binding, "tasks": len(rows), "error_type": type(exc).__name__, "error": str(exc)})
        raise


@torch.inference_mode()
def atomic_plan_metrics(model, tokenizer, token_ids, rows: list[dict], batch_size: int = 32):
    device = next(model.parameters()).device; counts = defaultdict(lambda: [0, 0]); total = 0; raw_predictions = []
    accounting = {"model_forward_passes": 0, "prompt_sequences": 0, "input_tokens": 0}
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]; prompts = []
        for row in chunk:
            op = row.get("operation") or row["witness"][0]
            target = trajectory(row["start"], [op], row["p"])[-1]
            prompts.append(plan_prompt({**row, "depth": 1, "target": list(target)}))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        accounting["model_forward_passes"] += 1; accounting["prompt_sequences"] += len(chunk)
        accounting["input_tokens"] += int(encoded.attention_mask.sum())
        positions = encoded.attention_mask.long().cumsum(-1) - 1; positions.masked_fill_(encoded.attention_mask == 0, 0)
        logits = model(**encoded, position_ids=positions).logits[:, -1]
        candidates = torch.tensor([token_ids[op] for op in OPS], device=device)
        predictions = logits[:, candidates].argmax(-1).tolist()
        for row, prediction in zip(chunk, predictions):
            op = row.get("operation") or row["witness"][0]; predicted_op = OPS[prediction]; passed = predicted_op == op
            counts[op][0] += passed; counts[op][1] += 1; total += passed
            raw_predictions.append({"task_id": row["task_id"], "operation": op,
                                    "plan_prediction": predicted_op, "plan_correct": bool(passed)})
    return ({"overall": total / len(rows), "by_operation": {op: passed / count for op, (passed, count) in counts.items()},
             "accounting": accounting}, raw_predictions)


@torch.inference_mode()
def atomic_apply_metrics(model, tokenizer, rows: list[dict], batch_size: int = 8, max_new_tokens: int = 24):
    device = next(model.parameters()).device; counts = defaultdict(lambda: [0, 0]); total = 0; generations = []
    accounting = {"generation_batches": 0, "generation_decode_steps": 0, "prompt_sequences": 0,
                  "prompt_tokens": 0, "decoded_completion_tokens": 0}
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]; prompts = []; targets = []
        for row in chunk:
            op = row.get("operation") or row["witness"][0]
            target = trajectory(row["start"], [op], row["p"])[-1]
            prompts.append(apply_prompt({**row, "program": [op]})); targets.append(format_state(target))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        accounting["generation_batches"] += 1; accounting["prompt_sequences"] += len(chunk)
        accounting["prompt_tokens"] += int(encoded.attention_mask.sum())
        prompt_width = encoded.input_ids.shape[1]
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(output[:, prompt_width:], skip_special_tokens=True)
        accounting["generation_decode_steps"] += int(output.shape[1] - prompt_width)
        accounting["decoded_completion_tokens"] += sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
        for row, target, text in zip(chunk, targets, texts):
            op = row.get("operation") or row["witness"][0]; passed = text.strip() == target
            counts[op][0] += passed; counts[op][1] += 1; total += passed
            generations.append({"schema": "stage4.distill.v5.atomic-generation.v1", "task_id": row["task_id"],
                                "operation": op, "target": target, "raw_completion": text, "correct": bool(passed),
                                "match_mode": "stripped_exact_equality"})
    return {"overall": total / len(rows), "by_operation": {op: passed / count for op, (passed, count) in counts.items()},
            "match_mode": "stripped_exact_equality", "accounting": accounting}, generations


def _derive_atomic_metrics(rows: list[dict], raw_rows: list[dict], branch: str, binding: dict):
    if len(raw_rows) != len(rows):
        raise RuntimeError("atomic raw row count mismatch")
    expected_ids = [row["task_id"] for row in rows]
    if len(set(expected_ids)) != len(expected_ids) or [row.get("task_id") for row in raw_rows] != expected_ids:
        raise RuntimeError("atomic raw task IDs/order mismatch")
    plan_counts = defaultdict(lambda: [0, 0]); apply_counts = defaultdict(lambda: [0, 0])
    plan_total = apply_total = 0
    for source, raw in zip(rows, raw_rows):
        op = source.get("operation") or source["witness"][0]
        target = format_state(trajectory(source["start"], [op], source["p"])[-1])
        expected = {"schema": "stage4.distill.v5.atomic-generation.v1", "task_id": source["task_id"],
                    "operation": op, "target": target, "match_mode": "stripped_exact_equality",
                    "branch": branch}
        for key, value in expected.items():
            if raw.get(key) != value: raise RuntimeError(f"atomic raw {source['task_id']} {key} mismatch")
        if not _binding_equal(raw.get("binding"), binding):
            raise RuntimeError(f"atomic raw {source['task_id']} binding mismatch")
        if "seed" in binding and raw.get("seed") != int(binding["seed"]):
            raise RuntimeError(f"atomic raw {source['task_id']} seed mismatch")
        prediction = raw.get("plan_prediction")
        if prediction not in OPS: raise RuntimeError(f"atomic raw {source['task_id']} PLAN prediction is invalid")
        plan_correct = prediction == op
        if not isinstance(raw.get("plan_correct"), bool) or raw["plan_correct"] != plan_correct:
            raise RuntimeError(f"atomic raw {source['task_id']} PLAN correctness mismatch")
        completion = raw.get("raw_completion")
        if not isinstance(completion, str): raise RuntimeError(f"atomic raw {source['task_id']} APPLY completion is invalid")
        apply_correct = completion.strip() == target
        if not isinstance(raw.get("correct"), bool) or raw["correct"] != apply_correct:
            raise RuntimeError(f"atomic raw {source['task_id']} APPLY correctness mismatch")
        plan_counts[op][0] += plan_correct; plan_counts[op][1] += 1; plan_total += plan_correct
        apply_counts[op][0] += apply_correct; apply_counts[op][1] += 1; apply_total += apply_correct
    missing = [op for op in OPS if plan_counts[op][1] == 0 or apply_counts[op][1] == 0]
    if missing: raise RuntimeError(f"atomic raw operation coverage is incomplete: {missing}")
    plan = {"overall": plan_total / len(rows),
            "by_operation": {op: plan_counts[op][0] / plan_counts[op][1] for op in OPS}}
    apply = {"overall": apply_total / len(rows),
             "by_operation": {op: apply_counts[op][0] / apply_counts[op][1] for op in OPS},
             "match_mode": "stripped_exact_equality"}
    return plan, apply


def _scientific_atomic(metric: dict, *, apply: bool) -> dict:
    result = {"overall": metric.get("overall"), "by_operation": metric.get("by_operation")}
    if apply: result["match_mode"] = metric.get("match_mode")
    return result


def evaluate_atomic(model, tokenizer, token_ids, rows, output_dir: Path, branch: str, binding: dict):
    output_dir = Path(output_dir); result_path = output_dir / "atomic" / f"{branch}.json"
    generation_path = output_dir / "atomic" / f"{branch}.generations.jsonl"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        valid = (result.get("schema") == "stage4.composition.v5.atomic-eval.v1" and result.get("status") == "DONE" and
                 result.get("branch") == branch and _binding_equal(result.get("binding"), binding) and
                 result.get("tasks") == len(rows) and result.get("generation_rows") == len(rows) and
                 generation_path.exists() and result.get("generation_sha256") == sha256(generation_path))
        if not valid: raise RuntimeError(f"mismatched cached atomic evaluation: {result_path}")
        raw_rows = read_jsonl(generation_path)
        derived_plan, derived_apply = _derive_atomic_metrics(rows, raw_rows, branch, binding)
        if (_scientific_atomic(result.get("plan", {}), apply=False) != derived_plan or
                _scientific_atomic(result.get("apply", {}), apply=True) != derived_apply):
            raise RuntimeError(f"cached atomic scalars differ from raw per-task evidence: {result_path}")
        plan_accounting = result["plan"].get("accounting")
        apply_accounting = result["apply"].get("accounting")
        if not isinstance(plan_accounting, dict) or not isinstance(apply_accounting, dict):
            raise RuntimeError(f"cached atomic evaluation lacks receipt accounting: {result_path}")
        return {**result, "plan": {**derived_plan, "accounting": plan_accounting},
                "apply": {**derived_apply, "accounting": apply_accounting}}
    started = time.time(); plan, plan_rows = atomic_plan_metrics(model, tokenizer, token_ids, rows)
    apply, generations = atomic_apply_metrics(model, tokenizer, rows)
    if [row["task_id"] for row in plan_rows] != [row["task_id"] for row in generations]:
        raise RuntimeError("fresh atomic PLAN/APPLY task order mismatch")
    for plan_row, generation in zip(plan_rows, generations):
        if plan_row["operation"] != generation["operation"]: raise RuntimeError("fresh atomic operation mismatch")
        generation.update({"plan_prediction": plan_row["plan_prediction"], "plan_correct": plan_row["plan_correct"],
                           "branch": branch, "binding": binding})
        if "seed" in binding: generation["seed"] = int(binding["seed"])
    derived_plan, derived_apply = _derive_atomic_metrics(rows, generations, branch, binding)
    if (_scientific_atomic(plan, apply=False) != derived_plan or
            _scientific_atomic(apply, apply=True) != derived_apply):
        raise RuntimeError("fresh atomic scalars differ from raw per-task evidence")
    plan = {**derived_plan, "accounting": plan["accounting"]}
    apply = {**derived_apply, "accounting": apply["accounting"]}
    write_jsonl(generation_path, generations)
    result = {"schema": "stage4.composition.v5.atomic-eval.v1", "status": "DONE", "branch": branch,
              "binding": binding, "plan": plan, "apply": apply, "wall_seconds": time.time() - started,
              "tasks": len(rows), "generation_rows": len(generations), "generation_sha256": sha256(generation_path)}
    dump_json(result_path, result)
    return result


def aligned_forgetting(base: dict, post: dict):
    drops = {"plan": base["plan"]["overall"] - post["plan"]["overall"]}
    for op in OPS:
        drops[f"plan_{op}"] = base["plan"]["by_operation"][op] - post["plan"]["by_operation"][op]
        drops[f"apply_{op}"] = base["apply"]["by_operation"][op] - post["apply"]["by_operation"][op]
    return {"drops": drops, "max_drop": max(drops.values()), "passes_le_002": max(drops.values()) <= 0.02 + 1e-12}


def unload(model):
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
ATOMIC_ROOT = REPO / "artifacts/stage4_sh1_v5_0p6b"
sys.path[:0] = [str(ROOT / "code"), str(ATOMIC_ROOT / "code")]

import v5_train as atomic_v5
from composition_core import OPS, TOKENS, apply_prompt, format_state, plan_prompt, program_answer, trajectory

PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    path = Path(path); digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"empty tree: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def tokenizer_and_ids(export_dir: Path | None = None):
    tokenizer = (AutoTokenizer.from_pretrained(str(export_dir), trust_remote_code=True)
                 if export_dir is not None else atomic_v5.v2.TOKENIZER)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    token_ids = {op: tokenizer.convert_tokens_to_ids(TOKENS[op]) for op in OPS}
    if len(set(token_ids.values())) != len(OPS):
        raise RuntimeError(f"operation token ids are not unique: {token_ids}")
    probe = tokenizer.encode("PROGRAM: " + " ".join(TOKENS[op] for op in OPS), add_special_tokens=False)
    if probe[-len(OPS):] != [token_ids[op] for op in OPS]:
        raise RuntimeError(f"operation tokens are not single-token in context: {probe[-10:]}")
    return tokenizer, token_ids


@torch.inference_mode()
def _probe_action_logits(model, tokenizer, token_ids, rows):
    device = next(model.parameters()).device
    prompts = []
    for row in rows:
        op = row["witness"][0]
        target = trajectory(row["start"], [op], row["p"])[-1]
        prompts.append(plan_prompt({**row, "depth": 1, "target": list(target)}))
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    positions = encoded.attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(encoded.attention_mask == 0, 0)
    logits = model(**encoded, position_ids=positions).logits[:, -1].float()
    return logits[:, [token_ids[op] for op in OPS]].cpu()


def export_atomic_checkpoint(decision_path: Path, output: Path, calibration_rows: list[dict], probe_n: int = 64):
    decision_path = Path(decision_path); output = Path(output)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    registered_pass = decision.get("status") == "PASS" and decision.get("composition_unlocked") is True
    exploratory_authorized = (
        decision.get("schema") == "stage4.distill.v5.exploratory-atomic-authorization.v1"
        and decision.get("status") == "AUTHORIZED_FOR_EXPLORATION"
        and decision.get("study_class") == "EXPLORATORY_FOLLOWUP"
        and decision.get("export_authorized") is True
        and decision.get("registered_atomic_pass") is False
        and decision.get("old_atomic_final_access_forbidden") is True
    )
    if not registered_pass and not exploratory_authorized:
        raise RuntimeError("atomic source is neither registered PASS nor explicitly authorized for exploration")
    selected = Path(decision["selected_adapter"])
    if not selected.is_absolute(): selected = REPO / selected
    if not selected.is_dir(): raise RuntimeError(f"selected atomic adapter is missing: {selected}")
    selected = selected.resolve(); selected_tree = tree_sha256(selected)
    if exploratory_authorized:
        canonical_authorization = (ROOT / "configs/exploratory_atomic_authorization.json").resolve()
        if decision_path.resolve() != canonical_authorization:
            raise RuntimeError("exploratory export requires the canonical authorization file")
        preregistration_path = ROOT / "manifests/preregistration.json"
        source_audit_path = ROOT / "manifests/exploratory_source_audit.json"
        if not preregistration_path.is_file() or not source_audit_path.is_file():
            raise RuntimeError("exploratory export requires preregistration and source-audit receipts")
        preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        required_source_names = {"atomic_decision", "calibration_gate", "calibration_generations",
                                 "training_receipt", "corrective_seal", "atomic_protocol", "data_manifest",
                                 "calibration_data", "corrective_data_manifest", "corrective_allocation",
                                 "corrective_sh1_data", "corrective_control_data", "corrective_plan_raw",
                                 "plan_raw_audit_receipt"}
        if set(decision.get("source_files", {})) != required_source_names:
            raise RuntimeError("exploratory authorization source set mismatch")
        if (decision.get("model") != PROTOCOL["model"] or
                str(preregistration.get("exploratory_atomic_authorization_sha256", "")).upper() != sha256(canonical_authorization).upper() or
                source_audit.get("status") != "PASS" or
                source_audit.get("registered_atomic_pass") is not False or
                source_audit.get("old_atomic_final_accessed") is not False or
                str(source_audit.get("authorization_sha256", "")).upper() != sha256(canonical_authorization).upper() or
                str(source_audit.get("selected_adapter_tree_sha256", "")).upper() != selected_tree.upper()):
            raise RuntimeError("exploratory export receipts are stale or inconsistent")
        expected_selected = (REPO / "artifacts/stage4_sh1_v5_0p6b/adapters/corrective").resolve()
        if selected != expected_selected or str(decision.get("selected_adapter_tree_sha256", "")).upper() != selected_tree.upper():
            raise RuntimeError("exploratory export source is not the authorized corrective checkpoint")
        for name, entry in decision["source_files"].items():
            source = Path(entry["path"])
            if not source.is_absolute(): source = REPO / source
            source = source.resolve()
            if REPO.resolve() not in source.parents or not source.is_file() or sha256(source).upper() != str(entry["sha256"]).upper():
                raise RuntimeError(f"exploratory export source mismatch: {name}")
        if ((ATOMIC_ROOT / "manifests/final_access_receipt.json").exists() or
                (ATOMIC_ROOT / "runs/generations/atomic_v5_final.jsonl").exists()):
            raise RuntimeError("old atomic final was opened; exploratory export is invalid")
    registered_decision_path = decision_path.resolve()
    if exploratory_authorized:
        registered_decision_path = Path(decision["source_files"]["atomic_decision"]["path"])
        if not registered_decision_path.is_absolute(): registered_decision_path = REPO / registered_decision_path
        registered_decision_path = registered_decision_path.resolve()
    source_authorization_path = decision_path.resolve() if exploratory_authorized else None
    receipt_path = output / "export_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {"model": PROTOCOL["model"], "atomic_decision_sha256": sha256(registered_decision_path),
                    "source_authorization_sha256": (sha256(source_authorization_path)
                                                     if source_authorization_path is not None else None),
                    "source_adapter_tree_sha256": selected_tree,
                    "source_authorization_schema": decision.get("schema"),
                    "registered_atomic_pass": bool(registered_pass)}
        if receipt.get("status") == "DONE" and all(receipt.get(k) == v for k, v in expected.items()) and receipt.get("export_tree_sha256") == _export_payload_sha(output):
            return receipt
        raise RuntimeError("mismatched existing atomic export")
    if output.exists(): raise RuntimeError(f"refusing to overwrite atomic export: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists(): raise RuntimeError(f"partial atomic export requires audit: {partial}")
    partial.mkdir(parents=True)
    tokenizer, token_ids = tokenizer_and_ids()
    model, _ = atomic_v5.v2.load_continuation(selected)
    # The registered atomic evaluator promotes this continuation to FP32 before
    # measuring it.  Merging while the frozen base is still BF16 changes action
    # logits by as much as 0.5, so promote first and preserve the evaluated
    # function.  This is a representation fix, not a source-checkpoint update.
    model.float(); model.eval()
    before = _probe_action_logits(model, tokenizer, token_ids, calibration_rows[:probe_n])
    merged = model.merge_and_unload(safe_merge=True); merged.tie_weights(); merged.eval()
    after_merge = _probe_action_logits(merged, tokenizer, token_ids, calibration_rows[:probe_n])
    merge_max_abs = float((before - after_merge).abs().max())
    merge_argmax_equal = bool(torch.equal(before.argmax(-1), after_merge.argmax(-1)))
    merged.save_pretrained(partial, safe_serialization=True); tokenizer.save_pretrained(partial)
    del model, merged; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(partial), dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True)
    if torch.cuda.is_available(): reloaded.cuda()
    reloaded.eval(); after_reload = _probe_action_logits(reloaded, tokenizer, token_ids, calibration_rows[:probe_n])
    reload_max_abs = float((after_merge - after_reload).abs().max())
    reload_argmax_equal = bool(torch.equal(after_merge.argmax(-1), after_reload.argmax(-1)))
    del reloaded; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not merge_argmax_equal or not reload_argmax_equal or merge_max_abs > 0.05 or reload_max_abs > 0.05:
        raise RuntimeError(f"atomic export equivalence failed: merge={merge_max_abs} reload={reload_max_abs}")
    receipt = {
        "schema": "stage4.distill.v5.atomic-export.v1", "status": "DONE", "model": PROTOCOL["model"],
        "atomic_decision": str(registered_decision_path),
        "atomic_decision_sha256": sha256(registered_decision_path),
        "source_authorization": str(source_authorization_path) if source_authorization_path is not None else None,
        "source_authorization_sha256": (sha256(source_authorization_path)
                                          if source_authorization_path is not None else None),
        "source_authorization_schema": decision.get("schema"),
        "study_class": decision.get("study_class", "REGISTERED"),
        "registered_atomic_pass": bool(registered_pass),
        "source_adapter": str(selected), "source_adapter_tree_sha256": selected_tree,
        "probe_rows": min(probe_n, len(calibration_rows)), "safe_merge": True,
        "source_evaluation_dtype": "torch.float32", "export_storage_dtype": "torch.float32",
        "merge_max_abs_logit_delta": merge_max_abs,
        "reload_max_abs_logit_delta": reload_max_abs, "merge_argmax_equal": merge_argmax_equal,
        "reload_argmax_equal": reload_argmax_equal, "token_ids": token_ids,
    }
    dump_json(partial / "export_receipt.json", receipt)
    partial.replace(output)
    # The receipt is excluded from its own tree fingerprint to avoid a recursive hash.
    export_files_hash = hashlib.sha256()
    for item in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "export_receipt.json"):
        export_files_hash.update(item.relative_to(output).as_posix().encode()); export_files_hash.update(b"\0"); export_files_hash.update(bytes.fromhex(sha256(item)))
    receipt["export_tree_sha256"] = export_files_hash.hexdigest()
    dump_json(output / "export_receipt.json", receipt)
    return receipt


def _export_payload_sha(output: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in Path(output).rglob("*") if p.is_file() and p.name != "export_receipt.json"):
        digest.update(item.relative_to(output).as_posix().encode()); digest.update(b"\0"); digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def require_bf16_cuda_training(resolved: dict) -> dict:
    if resolved.get("dtype") != "bfloat16":
        raise RuntimeError(f"registered Stage 4 training dtype must be bfloat16, got {resolved.get('dtype')!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("registered Stage 4 scientific training requires CUDA; CPU fallback is forbidden")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("registered Stage 4 scientific training requires native CUDA bf16 support")
    return {"runtime_device": "cuda", "runtime_dtype": "torch.bfloat16",
            "runtime_compute_dtype": "torch.bfloat16", "runtime_master_dtype": "torch.float32",
            "runtime_autocast_enabled": True,
            "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda}


def load_frozen_atomic(export_dir: Path, trainable: bool):
    export_dir = Path(export_dir)
    receipt = json.loads((export_dir / "export_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "DONE" or receipt.get("model") != PROTOCOL["model"] or receipt.get("export_tree_sha256") != _export_payload_sha(export_dir):
        raise RuntimeError("atomic export receipt/hash mismatch")
    tokenizer, token_ids = tokenizer_and_ids(export_dir)
    if trainable:
        require_bf16_cuda_training(PROTOCOL["pilot_initial"])
    # Keep the exact FP32 merged atomic function as the frozen master weights.
    # Training still uses registered BF16 compute through autocast below.
    model = AutoModelForCausalLM.from_pretrained(
        str(export_dir), dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True)
    if trainable:
        cfg = PROTOCOL["pilot_initial"]["lora"]
        lora = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"], bias="none",
                          task_type="CAUSAL_LM", target_modules=cfg["target_modules"],
                          trainable_token_indices=sorted(token_ids.values()), ensure_weight_tying=True)
        model = get_peft_model(model, lora)
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    if torch.cuda.is_available(): model.cuda()
    return model, tokenizer, token_ids


def load_branch(export_dir: Path, branch_adapter: Path | None):
    model, tokenizer, token_ids = load_frozen_atomic(export_dir, False)
    if branch_adapter is not None:
        model = PeftModel.from_pretrained(model, str(branch_adapter), is_trainable=False)
    model.float(); model.eval()
    return model, tokenizer, token_ids


def encode_pair(tokenizer, prompt: str, answer: str, *, kind: str, operation: str | None = None, max_length: int = 1024):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer + tokenizer.eos_token, add_special_tokens=False)
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise RuntimeError(f"example exceeds max_length={max_length}: {len(prompt_ids) + len(answer_ids)}")
    ids = prompt_ids + answer_ids
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids,
            "position_ids": list(range(len(ids))), "target_tokens": len(answer_ids),
            "prompt_tokens": len(prompt_ids), "total_tokens": len(ids), "kind": kind,
            "operation": operation, "segments": 1}


def atomic_example_specs(rows: list[dict]):
    specs = []
    for row in rows:
        op = row.get("operation") or row["witness"][0]
        target = trajectory(row["start"], [op], row["p"])[-1]
        normalized = {**row, "program": [op], "witness": [op], "depth": 1, "target": list(target)}
        specs.append((plan_prompt(normalized), program_answer([op]), "atomic_plan", op))
        specs.append((apply_prompt(normalized), format_state(target), "atomic_apply", op))
    return specs


def distill_example_specs(trajectories: list[dict], atomic_rows: list[dict], replay: float, seed: int):
    rng = random.Random(seed); specs = []
    for row in trajectories:
        answer = program_answer(row["program"]) + "\nTRACE: " + " -> ".join(format_state(state) for state in row["states"][1:])
        specs.append((plan_prompt({**row, "depth": len(row["program"])}), answer, "composition_distill", row["program"][0]))
    atomic = atomic_example_specs(atomic_rows)
    by_key = defaultdict(list)
    for spec in atomic: by_key[(spec[2], spec[3])].append(spec)
    replay_n = round(len(specs) * replay / (1.0 - replay))
    keys = sorted(by_key)
    for index in range(replay_n):
        key = keys[index % len(keys)]
        specs.append(rng.choice(by_key[key]))
    rng.shuffle(specs)
    return specs


def encode_specs(tokenizer, specs):
    return [encode_pair(tokenizer, prompt, answer, kind=kind, operation=op) for prompt, answer, kind, op in specs]


def pack_control_to_budget(control_encoded: list[dict], distill_encoded: list[dict], seed: int, max_length: int = 1024):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for row in control_encoded: pools[(row["kind"], row["operation"])].append(row)
    keys = sorted(pools); key_offset = rng.randrange(len(keys)); packed = []; masked = 0; segment_total = 0
    for record_index, distill in enumerate(distill_encoded):
        target = distill["target_tokens"]; ids = []; labels = []; positions = []; used = 0; segments = 0; prompt_count = 0
        attempts = 0
        while used < target:
            key = keys[(key_offset + record_index + segments) % len(keys)]
            candidate = rng.choice(pools[key]); attempts += 1
            if len(ids) + len(candidate["input_ids"]) > max_length:
                fitting = [row for values in pools.values() for row in values if len(ids) + len(row["input_ids"]) <= max_length]
                if not fitting: raise RuntimeError(f"cannot pack control target={target} into max_length={max_length}")
                candidate = max(fitting, key=lambda row: row["target_tokens"])
            ids.extend(candidate["input_ids"]); labels.extend(candidate["labels"])
            positions.extend(candidate["position_ids"]); used += candidate["target_tokens"]; prompt_count += candidate["prompt_tokens"]; segments += 1
            if attempts > 1000: raise RuntimeError("control packing did not converge")
        excess = used - target
        if excess:
            active = [index for index, label in enumerate(labels) if label != -100]
            for index in active[-excess:]: labels[index] = -100
            masked += excess; used -= excess
        packed.append({"input_ids": ids, "labels": labels, "position_ids": positions,
                       "target_tokens": used, "prompt_tokens": prompt_count,
                       "total_tokens": len(ids), "kind": "packed_atomic_control", "operation": None,
                       "segments": segments})
        segment_total += segments
    if [row["target_tokens"] for row in packed] != [row["target_tokens"] for row in distill_encoded]:
        raise RuntimeError("per-record control/distill target budgets differ")
    manifest = {
        "schema": "stage4.distill.v5.equal-budget.v1", "records_each": len(packed),
        "loss_bearing_target_tokens_each": sum(row["target_tokens"] for row in packed),
        "control_prompt_tokens": sum(row["prompt_tokens"] for row in packed),
        "distill_prompt_tokens": sum(row["prompt_tokens"] for row in distill_encoded),
        "control_total_tokens": sum(row["total_tokens"] for row in packed),
        "distill_total_tokens": sum(row["total_tokens"] for row in distill_encoded),
        "control_atomic_segments": segment_total, "masked_control_target_tokens": masked,
        "per_record_target_budget_equal": True, "packing": "independent segments via reset position_ids",
    }
    return packed, distill_encoded, manifest


class EncodedRows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def collate_packed(tokenizer, batch):
    width = max(len(row["input_ids"]) for row in batch); input_ids = []; labels = []; positions = []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append([tokenizer.pad_token_id] * pad + row["input_ids"])
        labels.append([-100] * pad + row["labels"])
        # Padding is an isolated ignored segment; a reset to zero starts the first real segment.
        pad_positions = list(range(pad))
        positions.append(pad_positions + row["position_ids"])
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "position_ids": torch.tensor(positions)}


def packed_attention_runtime_guard(model) -> dict:
    """Prove that reset position IDs activate Transformers' block-diagonal mask.

    Transformers >=4.57 uses ``find_packed_sequence_indices`` when a training
    call supplies position IDs and deliberately omits a 2-D attention mask.
    Failing this guard is safer than silently allowing cross-example attention.
    """
    from packaging.version import Version
    from transformers import __version__ as transformers_version
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, find_packed_sequence_indices

    if Version(torch.__version__.split("+")[0]) < Version("2.6"):
        raise RuntimeError("packed block-diagonal attention requires torch>=2.6")
    implementation = getattr(model.config, "_attn_implementation", None)
    if implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        raise RuntimeError(f"attention implementation cannot construct packed masks: {implementation}")
    if implementation not in {"sdpa", "eager", "flex_attention"}:
        raise RuntimeError(f"attention implementation does not honor packed mask functions: {implementation}")
    probe = torch.tensor([[0, 1, 2, 0, 1, 0]], dtype=torch.long)
    observed = find_packed_sequence_indices(probe).tolist()[0]
    expected = [0, 0, 0, 1, 1, 2]
    if observed != expected:
        raise RuntimeError(f"packed-sequence detection mismatch: {observed} != {expected}")
    return {"status": "PASS", "torch": torch.__version__, "transformers": transformers_version,
            "attention_implementation": implementation, "probe_position_ids": probe.tolist()[0],
            "probe_segment_ids": observed}


def _training_status_path(output: Path) -> Path:
    try:
        identity = Path(output).resolve().relative_to(ROOT.resolve()).as_posix().replace("/", "__")
    except ValueError:
        identity = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:16]
    return ROOT / "runs/training" / f"{identity}.status.json"


def _repair_completed_training_status(output: Path, expected: dict) -> None:
    path = _training_status_path(output)
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema": "stage4.distill.v5.training-status.v1", "started_at_unix": None, **expected}
    status.update({"status": "DONE", "finished_at_unix": status.get("finished_at_unix", time.time()),
                   "adapter_tree_sha256": tree_sha256(output), "recovered_from_complete_payload": True})
    dump_json(path, status)


def train_branch(export_dir: Path, output: Path, encoded: list[dict], resolved: dict, branch: str, input_hash: str):
    output = Path(output); receipt_path = output / "training_receipt.json"
    runtime = require_bf16_cuda_training(resolved)
    resolved_sha = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":"),
                                                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    target_tokens_per_epoch = sum(int(row["target_tokens"]) for row in encoded)
    input_tokens_per_epoch = sum(int(row["total_tokens"]) for row in encoded)
    prompt_tokens_per_epoch = sum(int(row["prompt_tokens"]) for row in encoded)
    expected = {"branch": branch, "seed": resolved["seed"], "lr": resolved["lr"], "epochs": resolved["epochs"],
                "effective_batch": resolved["effective_batch"], "records": len(encoded), "input_sha256": input_hash,
                "atomic_export_sha256": _export_payload_sha(export_dir),
                "resolved_config_sha256": resolved_sha,
                "loss_bearing_target_tokens_per_epoch": target_tokens_per_epoch,
                "loss_bearing_target_tokens_expected": target_tokens_per_epoch * int(resolved["epochs"]),
                "input_tokens_per_epoch": input_tokens_per_epoch,
                "prompt_tokens_per_epoch": prompt_tokens_per_epoch, **runtime}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "DONE" and all(receipt.get(key) == value for key, value in expected.items()):
            _repair_completed_training_status(output, expected); return receipt
        raise RuntimeError(f"mismatched completed branch: {output}")
    if output.exists(): raise RuntimeError(f"partial branch output: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial_receipt = partial / "training_receipt.json"
        required_payload = ((partial / "adapter_config.json").is_file() and
                            ((partial / "adapter_model.safetensors").is_file() or (partial / "adapter_model.bin").is_file()) and
                            (partial / "tokenizer.json").is_file())
        if partial_receipt.is_file() and required_payload:
            receipt = json.loads(partial_receipt.read_text(encoding="utf-8"))
            if receipt.get("status") == "DONE" and all(receipt.get(key) == value for key, value in expected.items()):
                partial.replace(output)
                _repair_completed_training_status(output, expected)
                return receipt
        raise RuntimeError(f"incomplete or mismatched branch output requires audit: {partial}")
    seed_all(resolved["seed"]); model, tokenizer, _ = load_frozen_atomic(export_dir, True); model.train()
    first_parameter = next(model.parameters())
    if first_parameter.device.type != "cuda" or first_parameter.dtype != torch.float32:
        raise RuntimeError(f"training model runtime mismatch: device={first_parameter.device}, dtype={first_parameter.dtype}")
    trainable_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.requires_grad}
    if trainable_dtypes != {torch.float32}:
        raise RuntimeError(f"training master parameter dtype mismatch: {sorted(map(str, trainable_dtypes))}")
    packed_attention = packed_attention_runtime_guard(model)
    micro = resolved["micro_batch"]; effective = resolved["effective_batch"]
    if effective % micro: raise RuntimeError("effective batch must be divisible by micro batch")
    accumulation = effective // micro
    loader = DataLoader(EncodedRows(encoded), batch_size=micro, shuffle=True,
                        collate_fn=lambda rows: collate_packed(tokenizer, rows),
                        generator=torch.Generator().manual_seed(resolved["seed"]))
    steps_per_epoch = math.ceil(len(loader) / accumulation); total_steps = steps_per_epoch * resolved["epochs"]
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=resolved["lr"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(total_steps * resolved["warmup_ratio"])), total_steps)
    status_path = _training_status_path(output)
    status = {"schema": "stage4.distill.v5.training-status.v1", "status": "STARTED", **expected,
              "optimizer_steps_expected": total_steps, "started_at_unix": time.time()}
    dump_json(status_path, status); started = time.time(); optimizer.zero_grad(set_to_none=True)
    losses = []; optimizer_steps = 0; seen_target_tokens = 0
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    try:
        for _epoch in range(resolved["epochs"]):
            for batch_index, batch in enumerate(loader):
                seen_target_tokens += int((batch["labels"] != -100).sum())
                batch = {key: value.cuda() if torch.cuda.is_available() else value for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_loss = model(**batch).loss
                (raw_loss / accumulation).backward(); losses.append(float(raw_loss.detach()))
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), resolved["max_grad_norm"])
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
                    if optimizer_steps % 50 == 0:
                        print(json.dumps({"branch": branch, "step": optimizer_steps, "steps": total_steps,
                                          "loss": sum(losses[-50 * accumulation:]) / len(losses[-50 * accumulation:])}), flush=True)
        partial.mkdir(parents=True); model.save_pretrained(partial); tokenizer.save_pretrained(partial)
        receipt = {"schema": "stage4.distill.v5.training-receipt.v1", "status": "DONE", **expected,
                   "optimizer_steps": optimizer_steps, "optimizer_steps_expected": total_steps,
                   "loss_bearing_target_tokens_seen": seen_target_tokens, "mean_loss": sum(losses) / len(losses),
                   "input_tokens_seen": input_tokens_per_epoch * int(resolved["epochs"]),
                   "prompt_tokens_seen": prompt_tokens_per_epoch * int(resolved["epochs"]),
                   "forward_microbatches": len(loader) * int(resolved["epochs"]),
                   "wall_seconds": time.time() - started,
                   "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                   "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                   "packed_attention": packed_attention}
        dump_json(partial / "training_receipt.json", receipt); partial.replace(output)
        status.update({"status": "DONE", "finished_at_unix": time.time(), "adapter_tree_sha256": tree_sha256(output)})
        dump_json(status_path, status); return receipt
    except Exception as exc:
        status.update({"status": "FAILED", "finished_at_unix": time.time(), "error_type": type(exc).__name__, "error": str(exc)})
        dump_json(status_path, status); raise
    finally:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

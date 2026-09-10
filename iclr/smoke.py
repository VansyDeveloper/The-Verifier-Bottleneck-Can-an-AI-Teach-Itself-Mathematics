"""Check the full pipeline on an offline, randomly initialized tiny Qwen3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def create_tiny_model(path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite model: {path}")
    path.mkdir(parents=True, exist_ok=True)
    tokens = (["<unk>", "<pad>", "<eos>"] + [str(i) for i in range(33)] +
              "FIELD DEGREE_CAP START TARGET ALLOWED MAX_STEPS Return exactly one line PROGRAM RESULT TRACE "
              "SH1 SC2 REV AC1 AX1 [ ] , : ->".split())
    backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
                                       eos_token="<eos>", model_max_length=1024, padding_side="left",
                                       model_input_names=["input_ids", "attention_mask"])
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=1024, attention_dropout=0.,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                        tie_word_embeddings=True)
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"choose an empty --out directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "TOKENIZERS_PARALLELISM": "false"}

    def run(label, arguments):
        command = [sys.executable, "-m", *map(str, arguments)]
        print(label, flush=True)
        with (out / f"{label}.stdout.log").open("w") as stdout, (out / f"{label}.stderr.log").open("w") as stderr:
            try:
                subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"{label} failed; see {out / (label + '.stderr.log')}") from exc

    try:
        model = create_tiny_model(out / "tiny_qwen3")
        run("data", ["iclr.data", "--out", out / "data", "--smoke"])
        base = out / "atomic" / "checkpoint"
        run("init", ["iclr.train", "--init", "--model", model, "--data", out / "data", "--out", base.parent,
                     "--epochs", 1, "--device", "cpu", "--micro-batch", 1, "--effective-batch", 10,
                     "--prefix-batch", 8, "--num-examples", 10])
        run("baseline", ["iclr.evaluate", "--base", base, "--data", out / "data",
                         "--out", out / "baseline", "--device", "cpu", "--prefix-batch", 8])
        from iclr.run import build_jobs, comparison_entries, parser as queue_parser, write_config, write_results
        args = queue_parser().parse_args([
            "--recipe", "trace", "size", "set", "--model", str(model), "--base", str(base),
            "--data", str(out / "data"), "--output", str(out), "--seeds", "0", "--epochs", "1",
            "--num-examples", "10", "--effective-batch", "10", "--device", "cpu"])
        jobs = build_jobs(args)
        jobs = [job for job in jobs if not (job["method"] == "ce" and job["supervision"] == "program"
                                           and job["budget_mode"] == "examples")]
        for index, job in enumerate(jobs):
            path = out / "configs" / f"{index}.json"
            write_config(path, job)
            run(f"branch_{index}", ["iclr.train", "--config", path])
        run("baseline_resume", ["iclr.evaluate", "--base", base, "--data", out / "data",
                                "--out", out / "baseline", "--device", "cpu", "--prefix-batch", 8])
        run("train_resume", ["iclr.train", "--config", out / "configs" / "0.json"])
        budgets = []
        for directory in [base.parent, *(Path(job["output"]) for job in jobs)]:
            assert (directory / "DONE").is_file(), f"missing completed receipt: {directory}"
            reload_check = json.loads((directory / "reload_check.json").read_text())
            assert reload_check["max_absolute_difference"] <= reload_check["tolerance"]
            if directory != base.parent:
                budgets.append(json.loads((directory / "budget.json").read_text()))
        assert len({(b["optimizer_steps"], b["all_target_tokens"]) for b in budgets[:3]}) == 1
        assert len({(b["optimizer_steps"], b["example_exposures"]) for b in budgets[3:]}) == 1
        for label in ("baseline_resume", "train_resume"):
            assert "Already complete:" in (out / f"{label}.stdout.log").read_text()
        comparison = out / "comparison.json"
        write_config(comparison, comparison_entries(jobs, out))
        write_results(jobs, out / "results.csv", out / "baseline")
        run("analysis", ["iclr.analyze", "--manifest", comparison, "--output", out / "analysis"])
        (out / "DONE.json").write_text(json.dumps({
            "status": "PASS", "model": "random tiny Qwen3; 1 layer, hidden size 32",
            "offline": True, "scientific_result": False, "training_branches": len(jobs),
            "matched_ce_target_tokens": budgets[0]["all_target_tokens"],
            "completed_run_reuse_checked": True,
            "wall_seconds": time.monotonic() - started}, indent=2) + "\n")
        print(f"PASS: {out / 'DONE.json'}", flush=True)
    except Exception as exc:
        (out / "FAILED.json").write_text(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()

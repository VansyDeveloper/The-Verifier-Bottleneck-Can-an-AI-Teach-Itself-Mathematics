"""SFT on the frozen offline pool (see gen_pool.py) — the no-exploration arm of H1.

Launch (8 GPUs):
  accelerate launch --num_processes 8 training/sft_offline.py --pool pool_offline.jsonl
"""

import argparse
import json
import os

from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--pool", default="pool_offline.jsonl")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="ph4_offsft")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.pool)]
    ds = Dataset.from_list(rows)
    out_dir = os.path.join("runs", args.run_name)

    cfg = SFTConfig(
        output_dir=out_dir,
        run_name=args.run_name,
        seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=1,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to=["tensorboard"],
    )
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "final"))


if __name__ == "__main__":
    main()

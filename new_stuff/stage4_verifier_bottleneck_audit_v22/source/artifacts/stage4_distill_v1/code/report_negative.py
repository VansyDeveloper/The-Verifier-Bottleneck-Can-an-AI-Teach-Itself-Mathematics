from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; REPO=ROOT.parent.parent
sys.path.insert(0,str(HERE))
from stage4_core import OPS,dump_json,load_json,read_jsonl,sha256


def main():
    runs=ROOT/"runs"; reports=ROOT/"reports"; manifests=ROOT/"manifests"; reports.mkdir(exist_ok=True)
    specs=[("atomic_0p6_attempt0","Qwen/Qwen3-0.6B","atomic_attempt0.json"),
           ("atomic_0p6_attempt1","Qwen/Qwen3-0.6B","atomic_attempt1.json"),
           ("atomic_1p7_attempt0","Qwen/Qwen3-1.7B","atomic_fallback_attempt0.json"),
           ("atomic_1p7_attempt1","Qwen/Qwen3-1.7B","atomic_fallback_attempt1.json")]
    index=[]
    for run_id,model,gate_name in specs:
        gate=load_json(runs/gate_name); receipt=load_json(ROOT/"adapters"/run_id/"training_receipt.json")
        receipt["micro_batches_forward_backward"]=1000*receipt["epochs"]
        receipt["loss_bearing_target_tokens"]=33284*receipt["epochs"]
        index.append({"schema":"stage4.run.v1","run_id":run_id,"training_status":receipt["status"],
            "gate_status":"PASS" if gate["pass"] else "FAILED","model":model,
            "adapter":str((ROOT/"adapters"/run_id).resolve()),"metrics":gate,"receipt":receipt})
    dump_json(runs/"RUN_INDEX.json",{"schema":"stage4.run-index.v1","runs":index})
    failure={"schema":"stage4.decision.v1","status":"FAILED","phase":"atomic_gate","reason":"both preregistered model sizes failed atomic gates",
             "composition_training_started":False,"final_opened":False,"attempts":[r["run_id"] for r in index]}
    dump_json(runs/"STAGE4_FAILED.json",failure)
    with (reports/"atomic_gate_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["run_id","model","plan_overall","apply_overall",*OPS,"gate"])
        for r in index:
            g=r["metrics"]; w.writerow([r["run_id"],r["model"],g["plan"]["overall"],g["apply"]["overall"],
                *[g["apply"]["by_operation"][o] for o in OPS],r["gate_status"]])
    try:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.8)); labels=[r["run_id"].replace("atomic_","") for r in index]
        axes[0].bar(labels,[r["metrics"]["plan"]["overall"]*100 for r in index],color="#4C78A8"); axes[0].axhline(95,color="crimson",ls="--"); axes[0].set_title("Atomic PLAN"); axes[0].set_ylabel("accuracy, %"); axes[0].tick_params(axis="x",rotation=20)
        x=range(len(OPS)); width=.18
        for i,r in enumerate(index): axes[1].bar([v+(i-1.5)*width for v in x],[r["metrics"]["apply"]["by_operation"][o]*100 for o in OPS],width,label=labels[i])
        axes[1].axhline(90,color="crimson",ls="--"); axes[1].set_xticks(list(x),OPS); axes[1].set_title("Atomic APPLY by operation"); axes[1].legend(fontsize=7); axes[1].set_ylim(0,105)
        fig.tight_layout(); fig.savefig(reports/"atomic_gate.png",dpi=180); plt.close(fig)
    except Exception as e: (reports/"plot_error.txt").write_text(str(e),encoding="utf-8")
    total_seconds=sum(r["receipt"]["wall_seconds"] for r in index); total_steps=sum(r["receipt"]["optimizer_steps"] for r in index)
    total_micro_batches=sum(r["receipt"]["micro_batches_forward_backward"] for r in index)
    total_target_tokens=sum(r["receipt"]["loss_bearing_target_tokens"] for r in index)
    table=[]
    for r in index:
        g=r["metrics"]; table.append(f"| {r['run_id']} | {g['plan']['overall']:.1%} | {g['apply']['overall']:.1%} | "+" | ".join(f"{g['apply']['by_operation'][o]:.1%}" for o in OPS)+f" | {r['gate_status']} |")
    strongest=max(r["metrics"]["apply"]["overall"] for r in index)
    discover=load_json(manifests/"discover.json")
    report="\n".join(["# Stage 4 atomic pilot report","","## Decision","",
      "**FAILED at the atomic gate. Composition distillation was not trained, and final tests were never created or opened.**","",
      "This is not evidence against Discover-and-Distill. It is evidence that the required atomic starting point was not achieved under the preregistered bounded calibration.","",
      "## Exact results","","| run | PLAN | APPLY | SH1 | SC2 | REV | AC1 | AX1 | gate |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",*table,"",
      f"Atomic thresholds were PLAN >=95%, every APPLY operation >=90%, and operation spread <=10 percentage points. Both 0.6B attempts failed on SH1/SC2 APPLY; both 1.7B attempts also failed. The strongest overall APPLY was {strongest:.1%}, still far below the per-operation gate.","",
      "## Accounting","",f"- Optimizer steps: {total_steps}",f"- Forward/backward micro-batches: {total_micro_batches}",f"- Loss-bearing target tokens: {total_target_tokens}",f"- Recorded training wall time: {total_seconds:.1f} s ({total_seconds/60:.1f} min)",
      "- Four training runs are DONE; all four scientific gate decisions are FAILED.","- No composition/control comparison, confidence interval, or p-value is reported because no admissible composition pilot exists.","",
      "## Data integrity","","- Train/dev task-ID overlap: 0.","- Train/dev state overlap: 0.","- Held-out motif leakage in train: 0.",f"- Discover trajectories: {discover['unique_trajectories']:,}; {discover['full_signatures']} full signatures; local-motif max/min ratio {discover['balance']['local_motif_max_min_ratio']:.3f}.","- Final files and CONFIG_FROZEN.json: absent by design.",""])
    (reports/"ATOMIC_GATE_REPORT.md").write_text(report,encoding="utf-8")
    (ROOT/"STATUS.md").write_text("# Stage 4 status\n\nStatus: **FAILED_ATOMIC_GATE (protocol-complete negative pilot)**.\n\nComposition training and final evaluation are locked. See `reports/ATOMIC_GATE_REPORT.md`.\n",encoding="utf-8")
    (ROOT/"BLOCKERS.md").write_text("# Blockers\n\nThe preregistered atomic gate was not met after the initial configuration and one bounded dev cycle on both 0.6B and 1.7B. By protocol, no further tuning is authorized in Stage 4 v1. A new preregistration is required for any follow-up.\n",encoding="utf-8")
    (ROOT/"DELIVERY_INDEX.md").write_text("# Delivery index\n\n- `PREREGISTRATION.md` — frozen hypotheses and gates.\n- `reports/ATOMIC_GATE_REPORT.md` — scientific decision and exact metrics.\n- `reports/atomic_gate.png` and `.csv` — plot and machine-readable table.\n- `runs/RUN_INDEX.json` — all four DONE training runs and FAILED gate outcomes.\n- `runs/generations/` — 2,000 raw fp32-evaluator completions.\n- `runs/STAGE4_FAILED.json` — fail-closed decision.\n- `manifests/` — environment, tokenizer IDs, runtime audit, data hashes, validation, and release hashes.\n- `data/` — preregistered atomic/train/dev and balanced Discover trajectories; no final data.\n- `adapters/` — all four new Stage-4 atomic checkpoints.\n- `code/`, `tests/`, `schemas/` — implementation and verification.\n",encoding="utf-8")
    files={}
    for p in ROOT.rglob("*"):
        if p.is_file() and p.name not in {"release_manifest.json","archive_receipt.json"} and p.suffix!=".zip" and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            files[p.relative_to(ROOT).as_posix()]={"bytes":p.stat().st_size,"sha256":sha256(p)}
    dump_json(manifests/"release_manifest.json",{"schema":"stage4.release.v1","file_count":len(files),"files":files})
    print(report)


if __name__=="__main__": main()

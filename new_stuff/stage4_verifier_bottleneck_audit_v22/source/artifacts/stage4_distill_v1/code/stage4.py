from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; REPO=ROOT.parent.parent
sys.path.insert(0,str(HERE))

from stage4_core import *

PROTOCOL=load_json(ROOT/"configs/protocol.json")
DATA=ROOT/"data"; RUNS=ROOT/"runs"; ADAPTERS=ROOT/"adapters"; REPORTS=ROOT/"reports"; MANIFESTS=ROOT/"manifests"
DELTA=Path(r"C:\Users\NS.PRODSTAR\Downloads\verifier_bottleneck_stage4_delta_instructions_only.zip")


def command_output(args): return subprocess.check_output(args,cwd=REPO,text=True,stderr=subprocess.STDOUT).strip()


def prepare():
    for d in (DATA,RUNS,ADAPTERS,REPORTS,MANIFESTS,ROOT/"rankings"): d.mkdir(parents=True,exist_ok=True)
    actual=sha256(DELTA)
    if actual != "02484C30E2FFE9D3DC91E7F22D44D0014C9D5AA6FBB8B35DD2227DAA3A091E08": raise RuntimeError("delta ZIP hash mismatch")
    versions={"python":sys.version,"platform":platform.platform(),"git_commit":command_output(["git","rev-parse","HEAD"]),
              "git_branch":command_output(["git","branch","--show-current"]),"delta_zip":str(DELTA),"delta_sha256":actual}
    for pkg in ("torch","transformers","peft","numpy","scipy"):
        try: versions[pkg]=__import__(pkg).__version__
        except Exception as e: versions[pkg]=f"unavailable:{e}"
    if __import__("torch").cuda.is_available(): versions["gpu"]=__import__("torch").cuda.get_device_name(0)
    environment_path=MANIFESTS/"environment.json"
    if not environment_path.exists() or not (RUNS/"STAGE4_FAILED.json").exists():
        dump_json(environment_path,versions)
    forbidden=set()
    atomic_train=generate_split("atomic_train",2000,PROTOCOL["train_fields"],[1],4100,"none",forbidden)
    atomic_dev=generate_split("atomic_dev",500,PROTOCOL["train_fields"],[1],4101,"none",forbidden)
    train=generate_split("train",4000,PROTOCOL["train_fields"],[2,3,4],4200,"none",forbidden)
    dev=generate_split("dev_a",500,PROTOCOL["train_fields"],[3],4201,"none",forbidden)
    splits={"atomic_train":atomic_train,"atomic_dev":atomic_dev,"train":train,"dev_a":dev}
    for name,rows in splits.items(): write_jsonl(DATA/f"{name}.jsonl",rows)
    audit=split_audit(splits); dump_json(MANIFESTS/"pre_final_split_audit.json",audit)
    if not audit["ok"]: raise RuntimeError(f"split audit failed: {audit}")
    traj,rep=discover(train); write_jsonl(DATA/"discover_trajectories.jsonl",traj); dump_json(MANIFESTS/"discover.json",rep)
    if not rep["passes_minimums"]: raise RuntimeError(f"discover minimum failed: {rep}")
    manifest={p.name: {"sha256":sha256(p),"bytes":p.stat().st_size,"rows":len(read_jsonl(p))} for p in DATA.glob("*.jsonl")}
    dump_json(MANIFESTS/"pre_final_data.json",manifest)
    print(json.dumps({"status":"DONE","audit":audit,"discover":rep},indent=2,default=dict))


def atomic_calibrate():
    from stage4_model import atomic_examples,encode_pair,stage4_tokenizer,train_branch,load_branch
    from stage4_eval import atomic_apply_accuracy,atomic_gate,atomic_plan_accuracy
    model_name=PROTOCOL["model_primary"]; source=REPO/PROTOCOL["source_adapter"]
    tok,tids=stage4_tokenizer(model_name); tok.save_pretrained(MANIFESTS/"tokenizer")
    dump_json(MANIFESTS/"operation_tokens.json",{"mapping":PROTOCOL["operation_tokens"],"token_ids":tids,"vocab_size":len(tok)})
    train=read_jsonl(DATA/"atomic_train.jsonl"); dev=read_jsonl(DATA/"atomic_dev.jsonl")
    encoded=[encode_pair(tok,p,a) for p,a,_,_ in atomic_examples(train)]
    configs=[{"lr":1e-4,"epochs":2},{"lr":2e-4,"epochs":3}]
    attempts=[]
    for i,cfg in enumerate(configs):
        prior=RUNS/f"atomic_attempt{i}.json"
        if prior.exists():
            gate=load_json(prior); attempts.append(gate)
            if gate.get("pass"): dump_json(MANIFESTS/"atomic_selected.json",gate); return gate
            continue
        out=ADAPTERS/f"atomic_0p6_attempt{i}"
        train_branch(model_name,source,out,tok,tids,encoded,seed=0,lr=cfg["lr"],epochs=cfg["epochs"],effective_batch=64)
        model=load_branch(model_name,source,out,tok,tids)
        plan=atomic_plan_accuracy(model,tok,tids,dev); apply=atomic_apply_accuracy(model,tok,dev)
        gate=atomic_gate(plan,apply); gate.update({"model":model_name,"config":cfg,"adapter":str(out.resolve())})
        dump_json(RUNS/f"atomic_attempt{i}.json",gate); attempts.append(gate)
        del model; __import__("torch").cuda.empty_cache()
        if gate["pass"]: dump_json(MANIFESTS/"atomic_selected.json",gate); return gate
    # The fallback is intentionally fail-closed; only download/use it after both bounded 0.6B attempts.
    model_name=PROTOCOL["model_fallback"]
    source=ROOT/"nonexistent_fallback_parent"
    tok,tids=stage4_tokenizer(model_name); encoded=[encode_pair(tok,p,a) for p,a,_,_ in atomic_examples(train)]
    for i,cfg in enumerate(configs):
        prior=RUNS/f"atomic_fallback_attempt{i}.json"
        if prior.exists():
            gate=load_json(prior)
            if gate.get("pass"): dump_json(MANIFESTS/"atomic_selected.json",gate); return gate
            continue
        out=ADAPTERS/f"atomic_1p7_attempt{i}"
        train_branch(model_name,source,out,tok,tids,encoded,seed=0,lr=cfg["lr"],epochs=cfg["epochs"],effective_batch=64)
        model=load_branch(model_name,source,out,tok,tids); plan=atomic_plan_accuracy(model,tok,tids,dev); apply=atomic_apply_accuracy(model,tok,dev)
        gate=atomic_gate(plan,apply); gate.update({"model":model_name,"config":cfg,"adapter":str(out.resolve())}); dump_json(RUNS/f"atomic_fallback_attempt{i}.json",gate)
        del model; __import__("torch").cuda.empty_cache()
        if gate["pass"]: dump_json(MANIFESTS/"atomic_selected.json",gate); return gate
    dump_json(RUNS/"STAGE4_FAILED.json",{"status":"FAILED","phase":"atomic_gate","attempts":attempts}); return {"pass":False}


def eval_branch(model_name,atomic_adapter,branch_adapter,split,branch):
    from stage4_model import stage4_tokenizer,load_branch
    from stage4_eval import evaluate_ranking
    if split.startswith("final_"):
        freeze=ROOT/"CONFIG_FROZEN.json"
        if not freeze.exists(): raise RuntimeError("final evaluator locked: CONFIG_FROZEN.json absent")
        record=load_json(freeze)["test_files"].get(split)
        if not record or sha256(DATA/f"{split}.jsonl") != record["sha256"]: raise RuntimeError(f"final evaluator hash guard failed: {split}")
    tok,tids=stage4_tokenizer(model_name); model=load_branch(model_name,Path(atomic_adapter),Path(branch_adapter) if branch_adapter else None,tok,tids)
    result=evaluate_ranking(model,tok,tids,read_jsonl(DATA/f"{split}.jsonl"),ROOT/"rankings"/split,branch)
    del model; __import__("torch").cuda.empty_cache(); return result


def pilot():
    if not (DATA/"train.jsonl").exists(): prepare()
    selected=load_json(MANIFESTS/"atomic_selected.json") if (MANIFESTS/"atomic_selected.json").exists() else atomic_calibrate()
    if not selected.get("pass"): print("Atomic gate failed; composition is locked."); return
    from stage4_model import stage4_tokenizer,atomic_examples,distill_examples,equalize_target_budget,train_branch
    model_name=selected["model"]; atomic=Path(selected["adapter"]); tok,tids=stage4_tokenizer(model_name)
    train=read_jsonl(DATA/"train.jsonl"); trajectories=read_jsonl(DATA/"discover_trajectories.jsonl"); atomic_rows=read_jsonl(DATA/"atomic_train.jsonl")
    control=atomic_examples(atomic_rows); distill=distill_examples(trajectories,atomic_rows,.2,0)
    ce,de,budget=equalize_target_budget(tok,control,distill,0)
    steps=max(1,__import__("math").ceil(len(de)/64)*2); budget.update({"optimizer_steps":steps,"effective_batch":64,"epochs":2})
    dump_json(MANIFESTS/"pilot_budget.json",budget)
    cpath=ADAPTERS/"pilot_seed0_atomic_control"; dpath=ADAPTERS/"pilot_seed0_composition_distill"
    train_branch(model_name,atomic,cpath,tok,tids,ce,seed=0,lr=1e-4,epochs=2,effective_batch=64,forced_steps=steps)
    train_branch(model_name,atomic,dpath,tok,tids,de,seed=0,lr=1e-4,epochs=2,effective_batch=64,forced_steps=steps)
    base,_=eval_branch(model_name,atomic,None,"dev_a","atomic_base")
    ctrl,cm=eval_branch(model_name,atomic,cpath,"dev_a","atomic_control")
    dist,dm=eval_branch(model_name,atomic,dpath,"dev_a","composition_distill")
    delta=dist["hit@32"]-ctrl["hit@32"]
    checks={"base_range":.05<=base["hit@32"]<=.50,"delta_ge_008":delta>=.08,
            "correct_mass_growth":dist["correct_mass"]>ctrl["correct_mass"],"leakage_zero":load_json(MANIFESTS/"pre_final_split_audit.json")["leakage_count"]==0,
            "atomic_gate":selected["pass"]}
    # Post-training atomic forgetting is measured before a pass decision.
    from stage4_model import load_branch
    from stage4_eval import atomic_plan_accuracy,atomic_apply_accuracy
    dev=read_jsonl(DATA/"atomic_dev.jsonl"); model=load_branch(model_name,atomic,dpath,tok,tids)
    postp=atomic_plan_accuracy(model,tok,tids,dev); posta=atomic_apply_accuracy(model,tok,dev); del model; __import__("torch").cuda.empty_cache()
    baseline=min([selected["plan"]["overall"]]+list(selected["apply"]["by_operation"].values()))
    after=min([postp["overall"]]+list(posta["by_operation"].values())); forgetting=max(0,baseline-after); checks["forgetting_le_002"]=forgetting<=.02
    decision={"status":"PASS" if all(checks.values()) else "FAILED","checks":checks,"delta_hit32":delta,"forgetting":forgetting,
              "base":base,"atomic_control":ctrl,"composition_distill":dist,"post_atomic":{"plan":postp,"apply":posta}}
    dump_json(RUNS/"pilot_decision.json",decision)
    if decision["status"]=="PASS": freeze_and_generate_final(selected,cpath,dpath)
    else: dump_json(RUNS/"STAGE4_FAILED.json",{"status":"FAILED","phase":"pilot_gate","decision":decision,"final_opened":False})
    print(json.dumps(decision,indent=2))


def freeze_and_generate_final(selected,cpath,dpath):
    occupied={(r["p"],tuple(s)) for p in (DATA/"atomic_train.jsonl",DATA/"atomic_dev.jsonl",DATA/"train.jsonl",DATA/"dev_a.jsonl") for r in read_jsonl(p) for s in r["states"]}
    specs=[("final_a",1000,PROTOCOL["train_fields"],"none",5001),("final_b",1000,PROTOCOL["train_fields"],"exactly_one",5002),
           ("final_c",500,PROTOCOL["transfer_fields"],"none",5003),("final_d",500,PROTOCOL["transfer_fields"],"exactly_one",5004)]
    hashes={}
    for name,n,fields,rule,seed in specs:
        rows=generate_split(name,n,fields,[3],seed,rule,occupied); path=DATA/f"{name}.jsonl"; write_jsonl(path,rows); hashes[name]={"sha256":sha256(path),"rows":n}
    all_paths=[DATA/f"{n}.jsonl" for n in ("atomic_train","atomic_dev","train","dev_a","final_a","final_b","final_c","final_d")]
    audit=split_audit({p.stem:read_jsonl(p) for p in all_paths})
    motif_checks={"final_b_exactly_one":all(r["motif_count"]==1 for r in read_jsonl(DATA/"final_b.jsonl")),
                  "final_d_exactly_one":all(r["motif_count"]==1 for r in read_jsonl(DATA/"final_d.jsonl"))}
    if not audit["ok"] or not all(motif_checks.values()): raise RuntimeError(f"final integrity failed: {audit}, {motif_checks}")
    dump_json(MANIFESTS/"final_split_audit.json",{"split_audit":audit,"motif_checks":motif_checks})
    frozen={"schema":"stage4.freeze.v1","selected_atomic":selected,"pilot_control":str(cpath.resolve()),"pilot_distill":str(dpath.resolve()),
            "protocol_sha256":sha256(ROOT/"configs/protocol.json"),"test_files":hashes,"frozen_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    dump_json(ROOT/"CONFIG_FROZEN.json",frozen)


def confirm():
    frozen_path=ROOT/"CONFIG_FROZEN.json"
    if not frozen_path.exists(): raise RuntimeError("final evaluator locked: CONFIG_FROZEN.json absent")
    frozen=load_json(frozen_path)
    for name,rec in frozen["test_files"].items():
        if sha256(DATA/f"{name}.jsonl") != rec["sha256"]: raise RuntimeError(f"final hash mismatch: {name}")
    from stage4_model import stage4_tokenizer,atomic_examples,distill_examples,equalize_target_budget,train_branch,load_branch
    from stage4_eval import atomic_plan_accuracy,atomic_apply_accuracy
    from stage4_stats import seed_statistics,exact_sign_flip,hierarchical_bootstrap
    selected=frozen["selected_atomic"]; model_name=selected["model"]; atomic=Path(selected["adapter"])
    tok,tids=stage4_tokenizer(model_name); atomic_rows=read_jsonl(DATA/"atomic_train.jsonl"); atomic_dev=read_jsonl(DATA/"atomic_dev.jsonl")
    trajectories=read_jsonl(DATA/"discover_trajectories.jsonl"); control_pool=atomic_examples(atomic_rows)
    cfg=PROTOCOL["pilot"]; all_seed={}; task_pairs={"final_a":{},"final_b":{}}
    for seed in PROTOCOL["confirm"]["seeds"]:
        cpath=ADAPTERS/f"confirm_seed{seed}_atomic_control"; dpath=ADAPTERS/f"confirm_seed{seed}_composition_distill"
        if seed==0 and Path(frozen["pilot_control"]).exists() and Path(frozen["pilot_distill"]).exists():
            cpath=Path(frozen["pilot_control"]); dpath=Path(frozen["pilot_distill"])
        else:
            distill=distill_examples(trajectories,atomic_rows,cfg["replay"],seed)
            ce,de,budget=equalize_target_budget(tok,control_pool,distill,seed)
            steps=max(1,__import__("math").ceil(len(de)/cfg["effective_batch"])*cfg["epochs"])
            budget.update({"optimizer_steps":steps,"effective_batch":cfg["effective_batch"],"epochs":cfg["epochs"]})
            dump_json(MANIFESTS/f"confirm_seed{seed}_budget.json",budget)
            train_branch(model_name,atomic,cpath,tok,tids,ce,seed=seed,lr=cfg["lr"],epochs=cfg["epochs"],effective_batch=cfg["effective_batch"],forced_steps=steps)
            train_branch(model_name,atomic,dpath,tok,tids,de,seed=seed,lr=cfg["lr"],epochs=cfg["epochs"],effective_batch=cfg["effective_batch"],forced_steps=steps)
        seed_result={}
        for split in ("final_a","final_b","final_c","final_d"):
            base,_=eval_branch(model_name,atomic,None,split,f"seed{seed}_atomic_base")
            ctrl,cm=eval_branch(model_name,atomic,cpath,split,f"seed{seed}_atomic_control")
            dist,dm=eval_branch(model_name,atomic,dpath,split,f"seed{seed}_composition_distill")
            seed_result[split]={"atomic_base":base,"atomic_control":ctrl,"composition_distill":dist,"delta_hit32":dist["hit@32"]-ctrl["hit@32"]}
            if split in task_pairs:
                task_pairs[split][seed]=[d["hit@32"]-c["hit@32"] for c,d in zip(cm,dm)]
        m=load_branch(model_name,atomic,dpath,tok,tids); pp=atomic_plan_accuracy(m,tok,tids,atomic_dev); pa=atomic_apply_accuracy(m,tok,atomic_dev)
        del m; __import__("torch").cuda.empty_cache(); seed_result["post_atomic"]={"plan":pp,"apply":pa}
        dump_json(RUNS/f"confirm_seed{seed}.json",seed_result); all_seed[seed]=seed_result
    stats={}
    for split in ("final_a","final_b"):
        diffs=[all_seed[s][split]["delta_hit32"] for s in PROTOCOL["confirm"]["seeds"]]
        ss=seed_statistics(diffs); ss["exact_sign_flip_p"]=exact_sign_flip(diffs)
        ss["hierarchical_bootstrap"]=hierarchical_bootstrap(task_pairs[split],PROTOCOL["confirm"]["bootstrap_repetitions"])
        mass_growth=all(all_seed[s][split]["composition_distill"]["correct_mass"]>all_seed[s][split]["atomic_control"]["correct_mass"] for s in all_seed)
        forgetting=[]
        baseline=min([selected["plan"]["overall"]]+list(selected["apply"]["by_operation"].values()))
        for s in all_seed:
            x=all_seed[s]["post_atomic"]; after=min([x["plan"]["overall"]]+list(x["apply"]["by_operation"].values())); forgetting.append(max(0,baseline-after))
        ss["criteria"]={"mean_delta_ge_005":ss["mean_delta"]>=.05,"ci_lower_gt_zero":ss["t_ci95"][0]>0,
            "t_p_lt_005":ss["p_two_sided"]<.05,"positive_at_least_5_of_6":ss["positive_seeds"]>=5,
            "forgetting_le_002":max(forgetting)<=.02,"correct_mass_growth":mass_growth}
        ss["positive_result"]=all(ss["criteria"].values()); ss["forgetting_by_seed"]=forgetting; stats[split]=ss
    dump_json(REPORTS/"confirmatory_statistics.json",stats)
    report=["# Stage 4 confirmatory report","",f"Final A positive: **{stats['final_a']['positive_result']}**",f"Final B positive: **{stats['final_b']['positive_result']}**","","## Statistics","","```json",json.dumps(stats,indent=2),"```",""]
    (REPORTS/"FINAL_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    dump_json(RUNS/"STAGE4_DONE.json",{"status":"DONE","confirmatory":True,"result_a":stats["final_a"]["positive_result"],"result_b":stats["final_b"]["positive_result"]})
    print(json.dumps(stats,indent=2))


def validate():
    errors=[]
    for path in ROOT.rglob("*.json"):
        try: json.loads(path.read_text(encoding="utf-8"))
        except Exception as e: errors.append(f"{path}: {e}")
    for path in ROOT.rglob("*.jsonl"):
        try: read_jsonl(path)
        except Exception as e: errors.append(f"{path}: {e}")
    try:
        import jsonschema
        task_schema=load_json(ROOT/"schemas/task.schema.json"); traj_schema=load_json(ROOT/"schemas/trajectory.schema.json"); run_schema=load_json(ROOT/"schemas/run.schema.json"); generation_schema=load_json(ROOT/"schemas/atomic_generation.schema.json")
        validated=0
        for path in DATA.glob("*.jsonl"):
            schema=traj_schema if path.name=="discover_trajectories.jsonl" else task_schema
            for row in read_jsonl(path): jsonschema.validate(row,schema); validated+=1
        index=load_json(RUNS/"RUN_INDEX.json")
        for row in index["runs"]: jsonschema.validate({k:v for k,v in row.items() if k!="schema"},run_schema); validated+=1
        for path in (RUNS/"generations").glob("*.jsonl"):
            for row in read_jsonl(path): jsonschema.validate(row,generation_schema); validated+=1
    except Exception as e: errors.append(f"schema validation: {e}"); validated=0
    failure=RUNS/"STAGE4_FAILED.json"
    if failure.exists():
        if (ROOT/"CONFIG_FROZEN.json").exists(): errors.append("negative atomic pilot must not have CONFIG_FROZEN.json")
        if list(DATA.glob("final_*.jsonl")): errors.append("negative atomic pilot must not contain final data")
    result={"status":"DONE" if not errors else "FAILED","errors":errors,"json_files":len(list(ROOT.rglob('*.json'))),"jsonl_files":len(list(ROOT.rglob('*.jsonl'))),"schema_rows_validated":validated}
    dump_json(MANIFESTS/"validation.json",result)
    if errors: raise RuntimeError(errors)
    print(json.dumps(result,indent=2))


def archive():
    validate(); full=ROOT/"stage4_distill_v1_complete.zip"; compact=ROOT/"stage4_distill_v1_report.zip"
    excluded={"__pycache__",".pytest_cache","cache"}
    files=[p for p in ROOT.rglob("*") if p.is_file() and not (set(p.parts)&excluded) and p.suffix!=".zip" and p.name!="archive_receipt.json"]
    dump_json(MANIFESTS/"archive_contents.json",{"schema":"stage4.archive-contents.v1","files":[p.relative_to(ROOT).as_posix() for p in files]})
    files=[p for p in ROOT.rglob("*") if p.is_file() and not (set(p.parts)&excluded) and p.suffix!=".zip" and p.name!="archive_receipt.json"]
    for out,include_adapters in ((full,True),(compact,False)):
        if out.exists(): out.unlink()
        with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in files:
                rel=p.relative_to(ROOT)
                if not include_adapters and rel.parts[0] in {"adapters","rankings"}: continue
                z.write(p,Path(ROOT.name)/rel)
        with zipfile.ZipFile(out,"r") as z:
            bad=z.testzip()
            if bad: raise RuntimeError(f"archive CRC failure: {out}: {bad}")
    receipt={"schema":"stage4.archive.v1","archives":{p.name:{"path":str(p.resolve()),"sha256":sha256(p),"bytes":p.stat().st_size,"crc_test":"PASS"} for p in (full,compact)}}
    dump_json(MANIFESTS/"archive_receipt.json",receipt); print(json.dumps(receipt,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("prepare","test","atomic","pilot","confirm","validate","archive")); a=ap.parse_args()
    if a.command=="prepare": prepare()
    elif a.command=="test": subprocess.check_call([sys.executable,"-m","pytest",str(ROOT/"tests"),"-q"],cwd=REPO)
    elif a.command=="atomic": atomic_calibrate()
    elif a.command=="pilot": pilot()
    elif a.command=="confirm": confirm()
    elif a.command=="validate": validate()
    elif a.command=="archive": archive()
if __name__=="__main__": main()

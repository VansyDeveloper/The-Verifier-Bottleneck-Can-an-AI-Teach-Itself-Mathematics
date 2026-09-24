from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent.parent; sys.path.insert(0,str(ROOT/"code"))
from stage4_core import dump_json,read_jsonl,write_jsonl
from stage4_eval import atomic_apply_accuracy,atomic_gate,atomic_plan_accuracy
from stage4_model import load_branch,stage4_tokenizer


def main():
    specs=[("atomic_0p6_attempt0","Qwen/Qwen3-0.6B",REPO/"artifacts/adapters/sft_atomic_r32_pilot_aw4_cont","atomic_attempt0.json"),
           ("atomic_0p6_attempt1","Qwen/Qwen3-0.6B",REPO/"artifacts/adapters/sft_atomic_r32_pilot_aw4_cont","atomic_attempt1.json"),
           ("atomic_1p7_attempt0","Qwen/Qwen3-1.7B",ROOT/"nonexistent_fallback_parent","atomic_fallback_attempt0.json"),
           ("atomic_1p7_attempt1","Qwen/Qwen3-1.7B",ROOT/"nonexistent_fallback_parent","atomic_fallback_attempt1.json")]
    dev=read_jsonl(ROOT/"data/atomic_dev.jsonl"); out=[]
    for run_id,model_name,parent,gate_name in specs:
        started=time.time(); tok,tids=stage4_tokenizer(model_name); model=load_branch(model_name,parent,ROOT/"adapters"/run_id,tok,tids)
        plan=atomic_plan_accuracy(model,tok,tids,dev); apply,generations=atomic_apply_accuracy(model,tok,dev,return_generations=True)
        gate=atomic_gate(plan,apply); previous=json.loads((ROOT/"runs"/gate_name).read_text())
        gate.update({"model":model_name,"config":previous["config"],"adapter":str((ROOT/"adapters"/run_id).resolve()),
                     "evaluation_dtype":"float32","evaluation_wall_seconds":time.time()-started,"generation_file":f"generations/{run_id}.jsonl"})
        dump_json(ROOT/"runs"/gate_name,gate); write_jsonl(ROOT/"runs"/"generations"/f"{run_id}.jsonl",generations); out.append(gate)
        del model; __import__("torch").cuda.empty_cache()
    dump_json(ROOT/"manifests"/"atomic_reevaluation.json",{"schema":"stage4.atomic-reevaluation.v1","status":"DONE","runs":len(out),"all_failed":not any(x["pass"] for x in out)})
    print(json.dumps(out,indent=2))


if __name__=="__main__": main()

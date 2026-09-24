from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent.parent; sys.path.insert(0,str(ROOT/"code"))
from stage4_core import OPS,TOKENS,dump_json,generate_split,read_jsonl
from stage4_eval import score_task
from stage4_model import load_branch,stage4_tokenizer


def weight_of(module):
    if hasattr(module,"weight"): return module.weight
    if hasattr(module,"token_adapter") and hasattr(module.token_adapter,"base_layer"): return module.token_adapter.base_layer.weight
    if hasattr(module,"original_module"): return module.original_module.weight
    raise AttributeError(type(module))


def main():
    model_name="Qwen/Qwen3-1.7B"; adapter=ROOT/"adapters/atomic_1p7_attempt1"; parent=ROOT/"nonexistent_fallback_parent"
    tok,tids=stage4_tokenizer(model_name); model=load_branch(model_name,parent,adapter,tok,tids)
    inp_module=model.get_input_embeddings(); out_module=model.get_output_embeddings()
    inp_adapter=inp_module.token_adapter; out_adapter=out_module.token_adapter
    tied=(inp_adapter.base_layer.weight is out_adapter.base_layer.weight and
          inp_adapter.trainable_tokens_delta["default"] is out_adapter.trainable_tokens_delta["default"] and
          inp_adapter.trainable_tokens_original["default"].data_ptr()==out_adapter.trainable_tokens_original["default"].data_ptr())
    row=read_jsonl(ROOT/"data/dev_a.jsonl")[0]; row={**row,"depth":2}
    m1,r1=score_task(model,tok,tids,row,batch_size=1); m2,r2=score_task(model,tok,tids,row,batch_size=32)
    by_program_1={tuple(x["program"]):x["score"] for x in r1}; by_program_2={tuple(x["program"]):x["score"] for x in r2}
    diffs=[abs(by_program_1[p]-by_program_2[p]) for p in by_program_1]
    same_order=[a["program"] for a in r1]==[b["program"] for b in r2]
    a=generate_split("det",20,[5,7],[2,3],99,"none",set()); b=generate_split("det",20,[5,7],[2,3],99,"none",set())
    receipts=[]
    for p in sorted((ROOT/"adapters").glob("*/training_receipt.json")):
        receipts.append({"path":str(p.relative_to(ROOT)),"status":json.loads(p.read_text())["status"]})
    result={"schema":"stage4.runtime-audit.v1","status":"PASS","gpu_smoke":True,"model":model_name,
      "single_token_ids":tids,"context_probe":tok.encode("PROGRAM: <OP0> <OP1>",add_special_tokens=False)[-2:],
      "input_output_embeddings_tied":tied,"enumeration_count":len(r1),"batched_unbatched_same_order":same_order,
      "batched_unbatched_max_abs_score_difference":max(diffs),"deterministic_generation":a==b,"resume_receipts":receipts}
    checks=[len(set(tids.values()))==5,result["context_probe"]==[tids["SH1"],tids["SC2"]],tied,len(r1)==25,same_order,max(diffs)<1e-3,a==b,all(x["status"]=="DONE" for x in receipts)]
    result["checks"]=checks; result["status"]="PASS" if all(checks) else "FAILED"
    dump_json(ROOT/"manifests/runtime_audit.json",result); print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main()

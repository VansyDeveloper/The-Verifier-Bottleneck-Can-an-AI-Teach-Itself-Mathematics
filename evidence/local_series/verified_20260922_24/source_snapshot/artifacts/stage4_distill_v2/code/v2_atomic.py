from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig,PeftModel,get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForCausalLM,get_cosine_schedule_with_warmup

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent.parent; V1=REPO/"artifacts/stage4_distill_v1/code"
sys.path.insert(0,str(V1))
from stage4_core import OPS,TOKENS,apply_prompt,dump_json,format_state,load_json,plan_prompt,read_jsonl,sha256,trajectory,write_jsonl
from stage4_eval import atomic_apply_accuracy,atomic_gate,atomic_plan_accuracy
from stage4_model import Rows,collate,encode_pair,stage4_tokenizer

CFG=load_json(ROOT/"configs/atomic_protocol.json"); DATA=ROOT/"data"; ADAPTERS=ROOT/"adapters"; RUNS=ROOT/"runs"; MANIFESTS=ROOT/"manifests"
DEFINITIONS={
 "SH1":"<OP0> maps coefficients c to b_j = sum over i>=j of c_i * binomial(i,j), modulo FIELD.",
 "SC2":"<OP1> maps [c0,c1,c2,c3,c4] to [c0,2*c1,4*c2,8*c3,16*c4], modulo FIELD and truncated to DEGREE_CAP.",
 "REV":"<OP2> reverses the coefficient list.","AC1":"<OP3> adds one modulo FIELD to c0 only.","AX1":"<OP4> adds one modulo FIELD to c1 only."
}


def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def bucket(p,state): return int(hashlib.sha256(json.dumps([p,list(state)],separators=(",",":")).encode()).hexdigest()[:8],16)%10


def task_id(split,op,p,state): return "s4v2-"+hashlib.sha256(json.dumps([split,op,p,list(state)],separators=(",",":")).encode()).hexdigest()[:20]


def sample_rows(split,per_op,degree_counts,partition,seed,used=None,only_ops=OPS):
 rng=random.Random(seed); used=set() if used is None else used; rows=[]
 for op in only_ops:
  for degree,count in degree_counts.items():
   got=0; attempts=0
   while got<count:
    attempts+=1
    if attempts>count*10000: raise RuntimeError(f"generation exhausted {split} {op} degree={degree}")
    p=rng.choice(CFG["fields"]); state=tuple(rng.randrange(p) for _ in range(degree+1))
    if all(x==0 for x in state) or (bucket(p,state)==0)!=(partition=="dev"): continue
    key=(op,p,state)
    if key in used: continue
    used.add(key); target=trajectory(state,[op],p)[-1]
    rows.append({"schema":"stage4.v2.atomic-task.v1","task_id":task_id(split,op,p,state),"split":split,"p":p,
      "degree":degree,"start":list(state),"operation":op,"program":[op],"witness":[op],"target":list(target)})
    got+=1
 return rows,used


def prepare():
 for d in (DATA,ADAPTERS,RUNS,MANIFESTS): d.mkdir(parents=True,exist_ok=True)
 train_counts={1:1000,2:1500,3:1500,4:1000}; dev_counts={2:67,3:67,4:66}
 train,used=sample_rows("atomic_train",5000,train_counts,"train",41000,set())
 dev,used=sample_rows("atomic_dev",200,dev_counts,"dev",41001,used)
 hard_counts={2:3500,3:3500,4:3000}; hard,used=sample_rows("hard_cycle",10000,hard_counts,"train",41002,used,only_ops=("SH1","SC2"))
 write_jsonl(DATA/"atomic_train.jsonl",train); write_jsonl(DATA/"atomic_dev.jsonl",dev); write_jsonl(DATA/"hard_cycle.jsonl",hard)
 audit={"counts":{"train":len(train),"dev":len(dev),"hard":len(hard)},
  "train_by_operation":Counter(r["operation"] for r in train),"dev_by_operation":Counter(r["operation"] for r in dev),
  "hard_by_operation":Counter(r["operation"] for r in hard),"unique_keys":len(used),
  "train_dev_overlap":len({(r['operation'],r['p'],tuple(r['start'])) for r in train}&{(r['operation'],r['p'],tuple(r['start'])) for r in dev}),
  "train_hard_overlap":len({(r['operation'],r['p'],tuple(r['start'])) for r in train}&{(r['operation'],r['p'],tuple(r['start'])) for r in hard})}
 audit["ok"]=audit["train_dev_overlap"]==0 and audit["train_hard_overlap"]==0 and all(v==5000 for v in audit["train_by_operation"].values()) and all(v==200 for v in audit["dev_by_operation"].values())
 dump_json(MANIFESTS/"data_audit.json",audit)
 manifest={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size,"rows":len(read_jsonl(p))} for p in DATA.glob("*.jsonl")}; dump_json(MANIFESTS/"data_manifest.json",manifest)
 versions={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),"git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip(),
  "torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"seed":CFG["seed"]}
 dump_json(MANIFESTS/"environment.json",versions)
 if not audit["ok"]: raise RuntimeError(audit)
 print(json.dumps(audit,indent=2,default=dict))


def scaffold_prompt(row): return "OPERATION RULE: "+DEFINITIONS[row["operation"]]+"\n\n"+apply_prompt(row)


def plain_examples(rows,plan_replay=0.0,seed=0):
 out=[encode_pair(TOKENIZER,apply_prompt(r),format_state(r["target"])) for r in rows]
 rng=random.Random(seed); n=round(len(rows)*plan_replay)
 for r in rng.sample(rows,n): out.append(encode_pair(TOKENIZER,plan_prompt({**r,"depth":1}),TOKENS[r["operation"]]))
 rng.shuffle(out); return out


def scaffold_examples(rows): return [encode_pair(TOKENIZER,scaffold_prompt(r),format_state(r["target"])) for r in rows]


def semantic_initialize(model,tok,tids):
 emb=model.get_input_embeddings().weight
 receipt={}
 with torch.no_grad():
  for op in OPS:
   source=tok.encode(op,add_special_tokens=False); value=emb[source].float().mean(0).to(emb.dtype); emb[tids[op]].copy_(value)
   receipt[op]={"target_id":tids[op],"source_ids":source,"source_text":op}
 model.tie_weights(); return receipt


def load_fresh(trainable=True):
 dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
 model=AutoModelForCausalLM.from_pretrained(CFG["model"],dtype=dtype,trust_remote_code=True,low_cpu_mem_usage=True)
 source=REPO/CFG["source_adapter"]; model=PeftModel.from_pretrained(model,str(source)).merge_and_unload()
 init=semantic_initialize(model,TOKENIZER,TIDS)
 if trainable:
  lc=CFG["lora"]; config=LoraConfig(r=lc["r"],lora_alpha=lc["alpha"],lora_dropout=lc["dropout"],bias="none",task_type="CAUSAL_LM",
   target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],trainable_token_indices=sorted(TIDS.values()),ensure_weight_tying=True)
  model=get_peft_model(model,config)
 if torch.cuda.is_available(): model.cuda()
 return model,init


def load_continuation(path):
 model,init=load_fresh(False); model=PeftModel.from_pretrained(model,str(path),is_trainable=True)
 if torch.cuda.is_available(): model.cuda()
 return model,init


def train_phase(model,encoded,spec,name):
 seed_all(CFG["seed"]); micro=8; accum=spec["effective_batch"]//micro
 loader=DataLoader(Rows(encoded),batch_size=micro,shuffle=True,collate_fn=lambda b:collate(TOKENIZER,b),generator=torch.Generator().manual_seed(CFG["seed"])); steps=math.ceil(len(loader)/accum)*spec["epochs"]
 opt=AdamW((p for p in model.parameters() if p.requires_grad),lr=spec["lr"]); sched=get_cosine_schedule_with_warmup(opt,max(1,round(steps*spec["warmup_ratio"])),steps)
 model.train(); opt.zero_grad(set_to_none=True); started=time.time(); losses=[]; step=0; seen_tokens=0
 for epoch in range(spec["epochs"]):
  for bi,b in enumerate(loader):
   seen_tokens+=int((b["labels"]!=-100).sum()); b={k:v.cuda() if torch.cuda.is_available() else v for k,v in b.items()}
   raw=model(**b).loss; (raw/accum).backward(); losses.append(float(raw.detach()))
   if (bi+1)%accum==0 or bi+1==len(loader):
    torch.nn.utils.clip_grad_norm_(model.parameters(),spec["max_grad_norm"]); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
    if step%50==0: print(json.dumps({"phase":name,"step":step,"steps":steps,"loss":sum(losses[-50*accum:])/len(losses[-50*accum:])}),flush=True)
 receipt={"phase":name,"status":"DONE","examples":len(encoded),"epochs":spec["epochs"],"optimizer_steps":step,"loss_bearing_target_tokens":seen_tokens,
  "mean_loss":sum(losses)/len(losses),"wall_seconds":time.time()-started,"lr":spec["lr"],"effective_batch":spec["effective_batch"]}
 dump_json(RUNS/f"{name}_training.json",receipt); return receipt


def evaluate(model,run_id):
 gc.collect()
 if torch.cuda.is_available(): torch.cuda.empty_cache()
 model.float(); model.eval(); dev=read_jsonl(DATA/"atomic_dev.jsonl"); started=time.time()
 plan=atomic_plan_accuracy(model,TOKENIZER,TIDS,dev,batch_size=16)
 apply,generations=atomic_apply_accuracy(model,TOKENIZER,dev,batch_size=8,return_generations=True)
 gate=atomic_gate(plan,apply); gate.update({"run_id":run_id,"evaluation_dtype":"float32","wall_seconds":time.time()-started})
 write_jsonl(RUNS/"generations"/f"{run_id}.jsonl",generations); dump_json(RUNS/f"{run_id}_gate.json",gate); return gate


def evaluate_initial_checkpoint():
 model,_=load_continuation(ADAPTERS/"atomic_initial")
 gate=evaluate(model,"atomic_initial")
 print(json.dumps(gate,indent=2))
 return gate


def initial():
 if not (DATA/"atomic_train.jsonl").exists(): prepare()
 train=read_jsonl(DATA/"atomic_train.jsonl"); model,init=load_fresh(True); dump_json(MANIFESTS/"semantic_token_initialization.json",init)
 receipts=[train_phase(model,scaffold_examples(train),CFG["phase1"],"initial_phase1"),
           train_phase(model,plain_examples(train,CFG["phase2"]["plan_replay"],0),CFG["phase2"],"initial_phase2")]
 out=ADAPTERS/"atomic_initial"; out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); TOKENIZER.save_pretrained(out); dump_json(out/"training_receipt.json",{"status":"DONE","phases":receipts,"semantic_initialization":init})
 gate=evaluate(model,"atomic_initial"); print(json.dumps(gate,indent=2)); return gate


def hard_cycle():
 initial_gate=load_json(RUNS/"atomic_initial_gate.json")
 allowed=(initial_gate["plan"]["overall"]>=CFG["gates"]["plan"] and all(initial_gate["apply"]["by_operation"][o]>=CFG["gates"]["apply_each"] for o in ("REV","AC1","AX1")))
 if not allowed: raise RuntimeError("hard cycle not authorized: failure is not limited to SH1/SC2")
 hard=read_jsonl(DATA/"hard_cycle.jsonl"); replay=read_jsonl(DATA/"atomic_train.jsonl"); rng=random.Random(99); replay=rng.sample(replay,round(len(hard)*CFG["hard_cycle"]["all_operation_replay"]))
 encoded=plain_examples(hard+replay,0.0,99); model,init=load_continuation(ADAPTERS/"atomic_initial"); receipt=train_phase(model,encoded,CFG["hard_cycle"],"hard_cycle")
 out=ADAPTERS/"atomic_hard_cycle"; out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); TOKENIZER.save_pretrained(out); dump_json(out/"training_receipt.json",{"status":"DONE","phase":receipt,"parent":"atomic_initial"})
 gate=evaluate(model,"atomic_hard_cycle"); print(json.dumps(gate,indent=2)); return gate


def run():
 gate=load_json(RUNS/"atomic_initial_gate.json") if (RUNS/"atomic_initial_gate.json").exists() else initial()
 if gate["pass"]: selected="atomic_initial"
 else:
  gate=load_json(RUNS/"atomic_hard_cycle_gate.json") if (RUNS/"atomic_hard_cycle_gate.json").exists() else hard_cycle(); selected="atomic_hard_cycle" if gate["pass"] else None
 decision={"schema":"stage4.v2.atomic-decision.v1","status":"PASS" if gate["pass"] else "FAILED","selected_adapter":selected,"gate":gate,
  "composition_unlocked":bool(gate["pass"])}; dump_json(RUNS/"ATOMIC_DECISION.json",decision); print(json.dumps(decision,indent=2))


TOKENIZER,TIDS=stage4_tokenizer(CFG["model"])

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("prepare","initial","evaluate-initial","hard-cycle","run")); a=ap.parse_args()
 {"prepare":prepare,"initial":initial,"evaluate-initial":evaluate_initial_checkpoint,"hard-cycle":hard_cycle,"run":run}[a.command]()
if __name__=="__main__": main()

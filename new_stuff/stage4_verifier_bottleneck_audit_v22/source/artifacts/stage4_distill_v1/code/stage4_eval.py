from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch

from stage4_core import OPS, TOKENS, dump_json, enumerate_programs, metrics_from_scores, plan_prompt, trajectory, write_jsonl


@torch.inference_mode()
def score_task(model, tok, token_ids, row, batch_size=32):
    device=next(model.parameters()).device; depth=int(row["depth"]); base=plan_prompt(row)
    prefix_scores={():0.0}
    for level in range(depth):
        prefixes=sorted(prefix_scores)
        for at in range(0,len(prefixes),batch_size):
            chunk=prefixes[at:at+batch_size]
            texts=[base + (" " + " ".join(TOKENS[o] for o in p) if p else "") for p in chunk]
            enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(device)
            position_ids=enc.attention_mask.long().cumsum(-1)-1; position_ids.masked_fill_(enc.attention_mask==0,0)
            logits=model(**enc,position_ids=position_ids).logits
            lp=torch.log_softmax(logits[:,-1].float(),dim=-1)
            for i,prefix in enumerate(chunk):
                for op in OPS:
                    prefix_scores[prefix+(op,)]=prefix_scores[prefix]+float(lp[i,token_ids[op]])
        prefix_scores={p:s for p,s in prefix_scores.items() if len(p)==level+1}
    programs=enumerate_programs(depth); scores=[prefix_scores[p] for p in programs]
    correct=[trajectory(row["start"],p,row["p"])[-1]==tuple(row["target"]) for p in programs]
    m=metrics_from_scores(programs,scores,correct)
    ranked=sorted(({"program":list(p),"score":s,"correct":c} for p,s,c in zip(programs,scores,correct)),key=lambda x:(-x["score"],tuple(x["program"])))
    return {"task_id":row["task_id"],"split":row["split"],"depth":depth,"p":row["p"],
            **m.hit,"correct_mass":m.correct_mass,"best_rank":m.best_rank,"mrr":m.mrr,"log_gap":m.log_gap},ranked


def evaluate_ranking(model,tok,token_ids,rows,out_dir:Path,branch:str,batch_size=32):
    metrics=[]; ranking_rows=[]
    for i,row in enumerate(rows):
        met,ranked=score_task(model,tok,token_ids,row,batch_size)
        metrics.append(met); ranking_rows.append({"schema":"stage4.ranking.v1","branch":branch,"task_id":row["task_id"],"ranking":ranked})
    out_dir.mkdir(parents=True,exist_ok=True)
    write_jsonl(out_dir/f"{branch}.rankings.jsonl.gz",ranking_rows,gzip_output=True)
    write_jsonl(out_dir/f"{branch}.metrics.jsonl",metrics)
    keys=("hit@1","hit@8","hit@16","hit@32","hit@64","correct_mass","mrr","log_gap")
    summary={"branch":branch,"n_tasks":len(metrics)}
    for k in keys:
        vals=[x[k] for x in metrics if math.isfinite(x[k])]
        summary[k]=sum(vals)/len(vals) if vals else None
    dump_json(out_dir/f"{branch}.summary.json",summary); return summary,metrics


@torch.inference_mode()
def atomic_plan_accuracy(model,tok,token_ids,rows,batch_size=64):
    device=next(model.parameters()).device; correct=0; by=defaultdict(lambda:[0,0])
    for at in range(0,len(rows),batch_size):
        chunk=rows[at:at+batch_size]; prompts=[]
        for r in chunk:
            op=r["witness"][0]; end=trajectory(r["start"],[op],r["p"])[-1]
            prompts.append(plan_prompt({**r,"depth":1,"target":list(end)}))
        enc=tok(prompts,return_tensors="pt",padding=True,add_special_tokens=False).to(device)
        position_ids=enc.attention_mask.long().cumsum(-1)-1; position_ids.masked_fill_(enc.attention_mask==0,0)
        logits=model(**enc,position_ids=position_ids).logits
        candidates=torch.tensor([token_ids[o] for o in OPS],device=device)
        pred=logits[:,-1][:,candidates].argmax(-1).tolist()
        for r,j in zip(chunk,pred):
            op=r["witness"][0]; ok=OPS[j]==op; correct+=ok; by[op][0]+=ok; by[op][1]+=1
    return {"overall":correct/len(rows),"by_operation":{o:a/n for o,(a,n) in by.items()}}


@torch.inference_mode()
def atomic_apply_accuracy(model,tok,rows,batch_size=32,max_new_tokens=20,return_generations=False):
    device=next(model.parameters()).device; by=defaultdict(lambda:[0,0]); correct=0; generations=[]
    from stage4_core import apply_prompt, format_state
    for at in range(0,len(rows),batch_size):
        chunk=rows[at:at+batch_size]; prompts=[]; targets=[]
        for r in chunk:
            op=r["witness"][0]; rr={**r,"program":[op]}; prompts.append(apply_prompt(rr))
            targets.append(format_state(trajectory(r["start"],[op],r["p"])[-1]))
        enc=tok(prompts,return_tensors="pt",padding=True,add_special_tokens=False).to(device)
        prompt_width=enc.input_ids.shape[1]
        out=model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        texts=tok.batch_decode(out[:,prompt_width:],skip_special_tokens=True)
        for r,target,text in zip(chunk,targets,texts):
            op=r["witness"][0]; ok=text.strip().startswith(target); correct+=ok; by[op][0]+=ok; by[op][1]+=1
            generations.append({"schema":"stage4.atomic-generation.v1","task_id":r["task_id"],"operation":op,
                                "target":target,"raw_completion":text,"correct":bool(ok)})
    metrics={"overall":correct/len(rows),"by_operation":{o:a/n for o,(a,n) in by.items()}}
    return (metrics,generations) if return_generations else metrics


def atomic_gate(plan,apply):
    vals=[apply["by_operation"].get(o,0) for o in OPS]
    checks={"plan_ge_095":plan["overall"]>=.95,"apply_each_ge_090":min(vals)>=.90,"spread_le_010":max(vals)-min(vals)<=.10+1e-12}
    return {"pass":all(checks.values()),"checks":checks,"plan":plan,"apply":apply}

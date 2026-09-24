from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from stage4_core import OPS, TOKENS, apply_prompt, dump_json, format_state, plan_prompt, program_answer, trajectory


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def stage4_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    added = [AddedToken(TOKENS[o], lstrip=True, rstrip=False, normalized=False, special=True) for o in OPS]
    tok.add_special_tokens({"additional_special_tokens": added})
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    ids = {op: tok.convert_tokens_to_ids(TOKENS[op]) for op in OPS}
    if len(set(ids.values())) != 5 or any(len(tok.encode(" " + TOKENS[o], add_special_tokens=False)) != 1 for o in OPS):
        raise RuntimeError(f"operation tokens are not unique single tokens: {ids}")
    probe = tok.encode("PROGRAM: " + " ".join(TOKENS[o] for o in ("SH1","SC2")), add_special_tokens=False)
    if probe[-2:] != [ids["SH1"], ids["SC2"]]:
        raise RuntimeError(f"context tokenization failed: {probe[-4:]}")
    return tok, ids


def load_atomic_start(model_name: str, source_adapter: Path, token_ids: dict[str,int], tokenizer, trainable: bool):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    kwargs = {"torch_dtype":dtype,"trust_remote_code":True,"low_cpu_mem_usage":True}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    chain=[]; current=Path(source_adapter)
    while current.exists():
        chain.append(current)
        receipt=current/"training_receipt.json"
        if not receipt.exists(): break
        parent=json.loads(receipt.read_text(encoding="utf-8")).get("parent_adapter")
        if not parent: break
        current=Path(parent)
    for adapter in reversed(chain):
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()
    if trainable:
        cfg = LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            trainable_token_indices=sorted(token_ids.values()),ensure_weight_tying=True)
        model = get_peft_model(model, cfg)
    if torch.cuda.is_available(): model.cuda()
    return model


def load_branch(model_name: str, source_adapter: Path, branch_adapter: Path | None, tokenizer, token_ids):
    model = load_atomic_start(model_name, source_adapter, token_ids, tokenizer, False)
    if branch_adapter is not None:
        model = PeftModel.from_pretrained(model, str(branch_adapter), is_trainable=False)
    # Exact exhaustive ranking is evaluated in fp32.  BF16 GEMM changes last-token logits by
    # batch shape (observed up to 0.25 log-prob), which can change near-tied program order.
    model.float()
    model.eval()
    return model


class Rows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def encode_pair(tok, prompt: str, answer: str, max_length=512):
    p = tok.encode(prompt, add_special_tokens=False)
    a = tok.encode(" " + answer + tok.eos_token, add_special_tokens=False)
    ids = (p+a)[:max_length]; n = min(len(p),len(ids))
    return {"input_ids":ids,"labels":[-100]*n+ids[n:],"target_tokens":max(0,len(ids)-n)}


def collate(tok, batch):
    m=max(len(x["input_ids"]) for x in batch); ii=[]; ll=[]; aa=[]
    for x in batch:
        n=m-len(x["input_ids"]); ii.append([tok.pad_token_id]*n+x["input_ids"])
        ll.append([-100]*n+x["labels"]); aa.append([0]*n+[1]*len(x["input_ids"]))
    attention_mask=torch.tensor(aa)
    position_ids=attention_mask.long().cumsum(-1)-1
    position_ids.masked_fill_(attention_mask==0,0)
    return {"input_ids":torch.tensor(ii),"labels":torch.tensor(ll),"attention_mask":attention_mask,"position_ids":position_ids}


def atomic_examples(rows):
    out=[]
    for r in rows:
        pr={**r,"program":[r["witness"][0]],"depth":1}
        end=trajectory(r["start"],pr["program"],r["p"])[-1]
        out.append((plan_prompt({**pr,"target":list(end)}),program_answer(pr["program"]),"plan",pr["program"][0]))
        out.append((apply_prompt(pr),format_state(end),"apply",pr["program"][0]))
    return out


def distill_examples(rows, replay_rows, replay: float, seed: int):
    rng=random.Random(seed); out=[]
    for r in rows:
        ans=program_answer(r["program"]) + " TRACE: " + " -> ".join(format_state(s) for s in r["states"][1:])
        out.append((plan_prompt({**r,"depth":len(r["program"])}),ans,"distill",r["program"][0]))
    target=round(len(out)*replay/(1-replay))
    atoms=atomic_examples(replay_rows)
    out.extend(rng.choice(atoms) for _ in range(target))
    rng.shuffle(out); return out


def equalize_target_budget(tok, control, distill, seed):
    rng=random.Random(seed)
    de=[encode_pair(tok,p,a) for p,a,_,_ in distill]; budget=sum(x["target_tokens"] for x in de)
    ce=[]; masked=0
    # Pair every distill record with one container of independent depth-1 tasks.  This preserves
    # the same number of examples/steps/effective batch while matching loss tokens exactly.
    for d in de:
        prompts=[]; answers=[]; x=None
        while x is None or x["target_tokens"] < d["target_tokens"]:
            p,a,_,_=rng.choice(control); prompts.append(p); answers.append(a)
            bundle_prompt="INDEPENDENT ATOMIC TASKS:\n"+"\n---\n".join(prompts)+"\nReturn the independent answers in order:"
            bundle_answer="\n".join(f"ANSWER {i+1}: {a}" for i,a in enumerate(answers))
            x=encode_pair(tok,bundle_prompt,bundle_answer)
        excess=x["target_tokens"]-d["target_tokens"]
        if excess:
            active=[j for j,v in enumerate(x["labels"]) if v!=-100]
            for j in active[-excess:]: x["labels"][j]=-100
            x["target_tokens"]-=excess; masked+=excess
        ce.append(x)
    return ce,de,{"loss_bearing_target_tokens":budget,"control_examples":len(ce),"distill_examples":len(de),
                  "masked_control_tokens":masked,"per_example_target_budget_equal":True}


def train_branch(model_name, source_adapter, output, tok, token_ids, encoded, *, seed, lr, epochs, effective_batch=64,
                 micro_batch=4, warmup_ratio=.03, max_grad_norm=1.0, forced_steps=None):
    receipt_path=Path(output)/"training_receipt.json"
    if receipt_path.exists():
        prior=json.loads(receipt_path.read_text(encoding="utf-8"))
        expected={"seed":seed,"lr":lr,"epochs":epochs,"effective_batch":effective_batch,"parent_adapter":str(Path(source_adapter).resolve())}
        if prior.get("status")=="DONE" and all(prior.get(k)==v for k,v in expected.items()): return prior
        raise RuntimeError(f"refusing to overwrite mismatched completed run: {output}")
    seed_all(seed); model=load_atomic_start(model_name,source_adapter,token_ids,tok,True); model.train()
    grad_acc=max(1,effective_batch//micro_batch)
    loader=DataLoader(Rows(encoded),batch_size=micro_batch,shuffle=True,collate_fn=lambda b:collate(tok,b),generator=torch.Generator().manual_seed(seed))
    natural=math.ceil(len(loader)/grad_acc)*epochs; total_steps=forced_steps or natural
    opt=AdamW((p for p in model.parameters() if p.requires_grad),lr=lr)
    sched=get_cosine_schedule_with_warmup(opt,max(1,round(total_steps*warmup_ratio)),total_steps)
    started=time.time(); losses=[]; step=0; opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for bi,b in enumerate(loader):
            b={k:v.cuda() if torch.cuda.is_available() else v for k,v in b.items()}
            loss=model(**b).loss/grad_acc; loss.backward(); losses.append(float(loss)*grad_acc)
            if (bi+1)%grad_acc==0 or bi+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
                if step>=total_steps: break
        if step>=total_steps: break
    output.mkdir(parents=True,exist_ok=True); model.save_pretrained(output); tok.save_pretrained(output)
    receipt={"status":"DONE","seed":seed,"optimizer_steps":step,"epochs":epochs,"effective_batch":effective_batch,
             "lr":lr,"mean_loss":sum(losses)/len(losses),"wall_seconds":time.time()-started,
             "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
             "parent_adapter":str(Path(source_adapter).resolve())}
    dump_json(output/"training_receipt.json",receipt); del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return receipt

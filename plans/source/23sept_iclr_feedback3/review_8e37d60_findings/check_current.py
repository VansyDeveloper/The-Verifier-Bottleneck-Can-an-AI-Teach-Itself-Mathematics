"""Independent CPU-only snapshot, recovery, data, and queue checks.
No model loading, model inference, CUDA test, or training is performed.
"""
import argparse,ast,hashlib,json,math,sys,types,tempfile
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
a=parser.parse_args();root=a.repo.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
from iclr.common import code_hash,file_hash,verify_data
m=json.loads((root/'REVIEW_ARCHIVE_MANIFEST.json').read_text())
print(type(m['files']),str(m['files'])[:250])
checks={}
files=m['files']; errors=[]
if isinstance(files,list):
 for e in files:
  p=root/e['path'];expected=e['sha256']
  if file_hash(p)!=expected:errors.append(e['path'])
else:
 for p,e in files.items():
  actual=root/p
  if file_hash(actual)!=e['sha256']: errors.append(p)
  if 'bytes' in e and actual.stat().st_size!=e['bytes']:errors.append(p+':size')
assert not errors,errors
checks['archive']={'files_checked':len(files),'errors':errors,'code_hash':code_hash(),'matches_code_hash':code_hash()==m['code_hash']}
assert checks['archive']['matches_code_hash']
# Run submitted recovery tests with only their needed source definitions loaded.
# Imports/loading infrastructure are excluded; run() stops at a mocked completion receipt.
path=root/'iclr/research_train.py';tree=ast.parse(path.read_text())
names={'OBJECTIVES','DEFAULTS','LOG_NAMES'};functions={'snapshot_logs','restore_logs','complete_checkpoint_receipts','config_checked','run'}
nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name in functions]
module=types.ModuleType('iclr.research_train');module.__dict__.update(math=math,Path=Path,hashlib=hashlib,start_run=lambda cfg,kind: ({},{},True))
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
sys.modules['iclr.research_train']=module
import iclr
iclr.research_train=module
import pytest
result=pytest.main([str(root/'tests/test_research_resume_recovery.py'),'-q','--disable-warnings'])
assert result==0
checks['recovery']={'submitted_cpu_cases':8,'exit_code':int(result),'scope':'Actual recovery/config function AST; mocked start_run returns completed sentinel; no real model or optimizer continuation.'}
# Record exact task counts, presets, and dependencies from real generated plan.
q=json.loads((root/'outputs/review_stability/queue.json').read_text())
tr=[j for j in q['jobs'] if j['kind']=='train'];cfgs=[json.loads((root/j['config_path']).read_text()) for j in tr]
assert len(tr)==4 and {(c['learning_rate'],c['replay_weight']) for c in cfgs}=={(1e-4,0),(3e-5,0),(1e-4,.25),(3e-5,.25)}
assert all(c['max_steps']==128 and c['checkpoint_steps']==[0,8,16,32,64,128] and c['objective']=='ce' and c['seed']==0 for c in cfgs)
assert len([j for j in q['jobs'] if j['id'].endswith('_initial')])==1
checks['queue']={'jobs':len(q['jobs']),'train_jobs':len(tr),'initial_evaluations':1,'final_dev_evaluations':4,'conditions':[{'id':j['id'],'lr':c['learning_rate'],'replay':c['replay_weight'],'max_steps':c['max_steps'],'depends_on':j['depends_on']} for j,c in zip(tr,cfgs)],'executed':False}
d=root/'outputs/shared_inputs_v4/mask1';verify_data(d)
from iclr.research_data import verify_amendment
amp=root/'evidence/research_v5/amendments/mask1/manifest.json';verify_amendment(amp,d,file_hash(root/'plans/research_v5.json'))
mon=json.loads(amp.read_text())['monitor'];counts={p.name:sum(bool(l.strip()) for l in p.read_text().splitlines()) for p in d.glob('dev*.jsonl')}
monitor_rows=sum(len(mon[k]) for k in ['ordinary','crossed','train'])
checks['data']={'all_mask1_file_hashes_and_amendment_verified':True,'dev_file_rows':counts,'monitor_rows_per_checkpoint':monitor_rows,'monitor_atomic_tasks':len(mon['atomic']),'monitor_calls':4*6,'monitor_depth3_task_evaluations':4*6*monitor_rows,'supervised_composition_task_presentations':4*128*4,'full_evals_depth3_rows':sum(v for k,v in counts.items() if not k.startswith('dev4_') and k!='dev_atomic.jsonl'),'full_evals_depth4_rows':sum(v for k,v in counts.items() if k.startswith('dev4_'))}
checks['scope']='Archive-bound code/data review; pinned operator weights absent; no production training or inference; no cross-device determinism claim.'
(out/'CHECKS.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(checks,indent=2,ensure_ascii=False))

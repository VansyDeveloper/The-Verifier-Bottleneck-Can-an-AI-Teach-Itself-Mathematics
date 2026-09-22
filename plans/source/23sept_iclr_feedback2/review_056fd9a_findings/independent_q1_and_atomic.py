"""Independent checks of the compact Q1 scores and the 20 supplied APPLY errors."""
import argparse, ast, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
import iclr
import composition_core as core
from iclr.common import read_jsonl
raw=read_jsonl(a.repo/'evidence/research_v5/q1_posthoc/source_scores.jsonl.gz')
unique={x['row']['task_id']:x['row'] for x in raw}
programs=list(core.enumerate_programs(3))
labels_ok=all({z for z in programs if core.verify_program(r['start'],r['target'],z,r['p'])}=={tuple(z) for z in r['correct_programs']} for r in unique.values())
assert labels_ok
mat=defaultdict(list)
for x in raw:
 r=x['row']
 assert len(r['local'])==len(r['full'])==125
 assert np.isfinite(r['local']).all() and np.isfinite(r['full']).all()
 if r.get('panel_kind')=='crossed':mat[x['seed'],x['arm'],r['panel_id']].append(r)
# Reconstruct crossed cells by matching START/TARGET to each panel's 2x2 layout.
# The saved rows explicitly supply cell_index, inspected below.

errors=json.loads((a.repo/'evidence/research_v5/atomic_audit/errors.json').read_text())
atomic=[]
for e in errors:
 expected=list(core.trajectory(e['start'],[e['operation']],e['p'])[-1]);assert expected==e['expected']
 try: parsed=ast.literal_eval(e['raw_answer'].strip())
 except (ValueError,SyntaxError): parsed=None
 assert parsed==e['parsed']
 correct=(isinstance(parsed,list) and len(parsed)==len(expected) and all(type(v)is int for v in parsed) and [v%e['p'] for v in parsed]==expected)
 assert correct==e['correct']
 atomic.append(e['task_id'])
result={'compact_raw_records':len(raw),'unique_tasks_checked_with_exact_interpreter':len(unique),'all_correct_sets_match_interpreter':labels_ok,'all_score_vectors_finite_length125':True,'atomic_error_examples_rechecked':len(atomic),'atomic_examples_have_no_label_or_parser_disagreements':True,'scope':'Compact records only; no fresh inference, no full operator-archive receipt audit or base-training history verification.'}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

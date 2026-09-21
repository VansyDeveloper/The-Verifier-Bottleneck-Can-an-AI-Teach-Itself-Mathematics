"""Prior-invariant same-START target alignment; no model calls.
Input: a rankings.jsonl from iclr.upgrade_evaluate (or the same schema).
Only panels with exactly one correct program per target are scored.
This is a proposed additional diagnostic, not a recomputation of missing model results.

python target_alignment.py --rankings FILE --output OUT.json --score-key local_score
python target_alignment.py --self-test
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
from collections import defaultdict
import numpy as np

def measure(matrix: np.ndarray, atol: float = 1e-8) -> dict:
    s=np.asarray(matrix,dtype=np.float64)
    if s.ndim!=2 or s.shape[0]!=s.shape[1] or not 2<=len(s)<=7 or not np.isfinite(s).all():
        raise ValueError('Expected a finite square matrix of 2..7 targets and their distinct correct programs')
    # Row and column means cancel in every assignment. Center first for stability.
    s=s-s.mean(1,keepdims=True)-s.mean(0,keepdims=True)+s.mean()
    perms=list(itertools.permutations(range(len(s))))
    values=np.array([sum(s[i,j] for i,j in enumerate(perm)) for perm in perms])
    true=values[0] # itertools first is identity
    tied=np.isclose(values,values.max(),rtol=0,atol=atol)
    credit=float(1/tied.sum()) if tied[0] else 0.0
    cycle=np.array([s[i,i]+s[j,j]-s[i,j]-s[j,i] for i,j in itertools.combinations(range(len(s)),2)])
    pair_credit=float(np.mean((cycle>atol)+.5*(np.abs(cycle)<=atol)))
    return {'targets':len(s),'assignment_credit':credit,'assignment_margin':float(true-values[1:].max()),
            'pair_accuracy_tie_half':pair_credit,'cycle_contrasts':cycle.tolist(),
            'permutations':len(perms),'maximizers':int(tied.sum()),'tie_tolerance':atol}

def analyze(path: Path, key: str, atol: float) -> dict:
    panels=defaultdict(list)
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r.get('panel_id'):panels[r['panel_id']].append(r)
    output=[];skipped=[]
    for pid,rows in sorted(panels.items()):
        rows.sort(key=lambda r:r['task_id'])
        if len({(r['p'],tuple(r['start']),r['depth']) for r in rows})!=1 or len({tuple(r['target']) for r in rows})!=len(rows):
            raise ValueError(f'Invalid same-START panel {pid}')
        correct=[[tuple(x['program']) for x in r['ranking'] if x['correct']] for r in rows]
        if any(len(x)!=1 for x in correct):
            skipped.append({'panel_id':pid,'reason':'nonunique solution(s)'});continue
        programs=[c[0] for c in correct]
        if len(set(programs))!=len(programs):raise ValueError(f'Distinct targets cannot share a deterministic solution: {pid}')
        table=[{tuple(x['program']):float(x[key]) for x in r['ranking']} for r in rows]
        if any(len(t)!=5**r['depth'] for t,r in zip(table,rows)):
            raise ValueError('Incomplete ranking')
        matrix=np.array([[t[p] for p in programs] for t in table])
        output.append({'panel_id':pid,'family':rows[0]['family'],'score_key':key,
                       'task_ids':[r['task_id'] for r in rows],**measure(matrix,atol)})
    summaries=[]
    for fam in sorted({r['family'] for r in output}):
        group=[r for r in output if r['family']==fam]
        summaries.append({'family':fam,'panels':len(group),**{m:float(np.mean([r[m] for r in group])) for m in
                         ('assignment_credit','assignment_margin','pair_accuracy_tie_half')}})
    return {'scope':'diagnostic; no new inference; unique-solution panels only; paired comparison between arms must use identical panel IDs',
            'tie_rule':'half credit on zero cycle; 1/n_maximizers if correct assignment is tied best',
            'panels':output,'summary':summaries,'skipped':skipped}

def test():
    rng=np.random.default_rng(3);s=rng.normal(size=(4,4))
    a=measure(s);b=measure(s+rng.normal(size=(4,1))*100+rng.normal(size=(1,4))*100)
    assert a['assignment_credit']==b['assignment_credit']
    assert a['pair_accuracy_tie_half']==b['pair_accuracy_tie_half']
    np.testing.assert_allclose(a['cycle_contrasts'],b['cycle_contrasts'],atol=1e-12,rtol=0)
    blind=measure(np.broadcast_to(np.arange(4.),(4,4)))
    assert blind['assignment_credit']==1/24 and blind['pair_accuracy_tie_half']==.5
    good=measure(np.eye(4));assert good['assignment_credit']==1 and good['pair_accuracy_tie_half']==1
    print('PASS: additive-prior and row-offset invariance; static scores give 1/24 and 1/2; aligned scores succeed.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rankings',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--score-key',choices=['score','local_score','format_score'],default='local_score')
    p.add_argument('--atol',type=float,default=1e-8);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:test()
    else:
        if not args.rankings or not args.output:p.error('--rankings and --output required')
        args.output.write_text(json.dumps(analyze(args.rankings,args.score_key,args.atol),indent=2)+'\n')

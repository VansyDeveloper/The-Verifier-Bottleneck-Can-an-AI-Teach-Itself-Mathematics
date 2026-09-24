from __future__ import annotations

import itertools
import math
import random
from statistics import mean, stdev


def normal_cdf(x): return .5*(1+math.erf(x/math.sqrt(2)))


def seed_statistics(differences, alpha=.05):
    n=len(differences); m=mean(differences)
    if n<2: return {"n":n,"mean_delta":m,"t_ci95":[None,None],"t":None,"p_two_sided":None}
    se=stdev(differences)/math.sqrt(n); t=m/se if se else math.inf
    # Exact Student critical values for the preregistered n=6, normal approximation otherwise.
    critical={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571}.get(n-1,1.96)
    try:
        from scipy.stats import t as student_t
        p=2*student_t.sf(abs(t),df=n-1)
    except Exception:
        p=2*(1-normal_cdf(abs(t)))
    return {"n":n,"mean_delta":m,"t_ci95":[m-critical*se,m+critical*se],"t":t,"p_two_sided":p,
            "positive_seeds":sum(x>0 for x in differences),"positive_ge_5pp":sum(x>=.05 for x in differences)}


def exact_sign_flip(differences):
    observed=abs(mean(differences)); n=len(differences); extreme=0
    for signs in itertools.product((-1,1),repeat=n):
        if abs(mean([s*x for s,x in zip(signs,differences)])) >= observed-1e-15: extreme+=1
    return extreme/(2**n)


def hierarchical_bootstrap(seed_tasks, repetitions=20000, seed=20260727):
    rng=random.Random(seed); seeds=sorted(seed_tasks); vals=[]
    for _ in range(repetitions):
        sampled=[rng.choice(seeds) for _ in seeds]; diffs=[]
        for s in sampled:
            pairs=seed_tasks[s]; draw=[rng.choice(pairs) for _ in pairs]; diffs.append(mean(draw))
        vals.append(mean(diffs))
    vals.sort(); return {"repetitions":repetitions,"ci95":[vals[int(.025*repetitions)],vals[min(repetitions-1,int(.975*repetitions))]],"mean":mean(vals)}


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get); m=len(ordered); adjusted={}; running=0
    for i,k in enumerate(ordered):
        running=max(running,(m-i)*pvalues[k]); adjusted[k]=min(1.0,running)
    return adjusted

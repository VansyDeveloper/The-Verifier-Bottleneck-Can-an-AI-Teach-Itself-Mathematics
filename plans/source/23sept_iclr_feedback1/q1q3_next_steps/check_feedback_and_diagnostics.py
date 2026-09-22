#!/usr/bin/env python3
"""Replay the compact Q1/Q3 bundle and add explicitly exploratory diagnostics.

Usage:
  python check_feedback_and_diagnostics.py /path/to/feedback_Q1_Q3_20260923 --out checks.json
Requires Python 3.10+ and numpy. No models, GPU, or network calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anova(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell order 00, 01, 10, 11. Coefficients: mean, start, target, interaction."""
    require(d.shape == (4,) and np.isfinite(d).all(), 'Need four finite cell contrasts')
    basis = np.array([[1., 1., 1., 1.], [1., 1., -1., -1.],
                      [1., -1., 1., -1.], [1., -1., -1., 1.]])
    coeff = basis @ d / 4.
    return coeff, basis.T @ coeff


def math_checks() -> dict:
    y = np.array([1., -1., -1., 1.])
    before = np.array([3., 1., 1., 3.]); after = before - 2.
    a, reconstructed = anova(before); b, _ = anova(after)
    np.testing.assert_allclose(reconstructed, before)
    require(not (before * y > 0).all() and (after * y > 0).all(), 'Offset illustration failed')
    np.testing.assert_allclose(a[1:], b[1:])
    require(before @ y == after @ y == 4., 'Interaction must cancel static offsets')

    rng = np.random.default_rng(20260923)
    for _ in range(50):
        d = rng.normal(size=4)
        coef, recovered = anova(d)
        np.testing.assert_allclose(recovered, d, atol=1e-14)
        gap = min(d[0], d[3]) - max(d[1], d[2])
        shift = -(min(d[0], d[3]) + max(d[1], d[2])) / 2.
        require(bool(gap > 0) == bool(((d + shift) * y > 0).all()), 'Offset feasibility check failed')
    p0, p1, r0, r1 = rng.random((4, 100))
    between = (p1 - p0) * (r1 + r0) / 2.
    within = (r1 - r0) * (p1 + p0) / 2.
    np.testing.assert_allclose(between + within, p1*r1 - p0*r0, atol=1e-15)
    # Full-vocabulary CE = local-action CE + legal-action gate CE.
    logits = rng.normal(size=17)
    logp = logits - np.logaddexp.reduce(logits)
    legal = logp[:5]; logz = np.logaddexp.reduce(legal)
    local = legal - logz
    np.testing.assert_allclose(-legal, -local - logz, atol=1e-15)
    return {'synthetic_checks': 'PASS', 'scope': 'algebra only, not a model training test',
            'tests': ['four-cell additive decomposition/reconstruction', 'static bias changes strict joint without changing I',
                      'scalar-offset solvability margin', 'exact symmetric probability-mass decomposition',
                      'full-vocabulary/local-action/format CE identity']}


def positive_interaction(root: Path, draws: int = 10000) -> dict:
    rows = []
    with (root/'analysis/panel_metrics.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['queue'] == 'Q1' and row['policy'] == 'local' and row['kind'] == 'crossed' and row['family'] in ('B','D') and row['arm'] in ('atomic_control', 'composition'):
                rows.append(row)
    cells = defaultdict(dict)
    for row in rows:
        key = (row['family'], row['panel_id'])
        subkey = (int(row['seed']), row['arm'])
        require(subkey not in cells[key], 'Duplicate observation')
        cells[key][subkey] = float(row['interaction'])
    seeds = [0,1,2]
    expected = {(s,a) for s in seeds for a in ('atomic_control','composition')}
    require(len(cells) == 32, 'Expected 32 crossed panels across B/D')
    require(all(set(c) == expected for c in cells.values()), 'Unpaired seeds or arms')
    values = defaultdict(list); rates = defaultdict(list); changes = defaultdict(int)
    per_seed = defaultdict(list)
    for key in sorted(cells):
        c = cells[key]
        diffs = []
        for seed in seeds:
            before = float(c[seed, 'atomic_control'] > 1e-8)
            after = float(c[seed, 'composition'] > 1e-8)
            diffs.append(after-before); per_seed[seed].append(after-before)
            rates['atomic_control'].append(before); rates['composition'].append(after)
            changes[f'{int(before)}->{int(after)}'] += 1
        values[key[0]].append(float(np.mean(diffs)))
    require(all(len(values[f])==16 for f in ('B','D')), 'Unexpected panel counts')
    rng=np.random.default_rng(20260923)
    boots=[]
    for family in ('B','D'):
        x=np.asarray(values[family]); boots.append(x[rng.integers(len(x),size=(draws,len(x)))].mean(1))
    ci=np.quantile(np.mean(boots,axis=0),[.025,.975])
    return {'name':'P(crossed interaction I > 1e-8)',
            'status':'new post-hoc diagnostic; not part of the v4 confirmatory endpoint sequence',
            'control':float(np.mean(rates['atomic_control'])), 'treatment':float(np.mean(rates['composition'])),
            'delta':float(np.mean([np.mean(values[f]) for f in ('B','D')])),
            'ci95':ci.tolist(), 'bootstrap_draws':draws, 'bootstrap_seed':20260923,
            'bootstrap_unit':'whole panel, paired seeds averaged first; equal B/D weights',
            'scope':'conditional on these 3 continuation checkpoint pairs and this original mask',
            'per_seed_delta':{str(s):float(np.mean(v)) for s,v in per_seed.items()},
            'panel_seed_transitions_not_independent_replications':dict(changes),
            'minimum_abs_I':min(abs(v) for c in cells.values() for v in c.values()),
            'limitation':'recomputed from saved panel scalars, not from raw logits; no multiplicity correction'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); root=args.bundle.resolve()
    require((root/'check_primary.py').is_file(), 'Missing compact bundle checker')
    proc=subprocess.run([sys.executable,str(root/'check_primary.py')],cwd=root,text=True,capture_output=True)
    require(proc.returncode==0, f'Original checker failed:\n{proc.stdout}\n{proc.stderr}')
    report={'input_bundle_manifest_sha256':sha(root/'MANIFEST.json'),
            'original_check_primary_stdout':proc.stdout,
            'independently_replayed_scope':'34 file hashes and 35 paired endpoint contrasts from compact stored observations; no raw logits/weights/model inference',
            'math_checks':math_checks(), 'new_exploratory_diagnostic':positive_interaction(root),
            'source_hashes':{name:sha(root/name) for name in ['analysis/panel_metrics.csv','analysis/atomic_retention.csv','analysis/training_audit.json','protocol/plans/research_v4.json']}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(proc.stdout,end=''); print(json.dumps(report['new_exploratory_diagnostic'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

# Mode-aware re-grading of the APPLY retention check

The registered check accepts only `RESULT: [c0, ..., cd]`. This table re-reads the
same stored generations and classifies each answer by **what it contains** rather
than by its prefix, because these adapters emit an operation sequence under both
the `RESULT:` and the `PROGRAM:` prefix. No model was run; no registered report
was modified.

| adapter | n | registered pass@1 | content: state vector | of those correct | content: a program | of those a correct plan | content: other |
|---|---:|---:|---:|---:|---:|---:|---:|
| atomic | 80 | 80 | 80 | 80 | 0 | 0 | 0 |
| oracle0 | 80 | 0 | 0 | 0 | 80 | 50 | 0 |
| oracle1 | 80 | 0 | 0 | 0 | 80 | 39 | 0 |
| oracle2 | 80 | 0 | 0 | 0 | 79 | 10 | 1 |
| nomotif0 | 80 | 0 | 0 | 0 | 80 | 43 | 0 |
| nomotif1 | 80 | 0 | 0 | 0 | 80 | 58 | 0 |
| nomotif2 | 80 | 0 | 0 | 0 | 80 | 40 | 0 |

## Reading

Across the 6 composition adapters (480 graded answers):
- 0/480 answers actually contain a coefficient vector, which is what an APPLY
  prompt asks for.
- 479/480 contain an operation sequence instead - the answer to the PLAN version
  of the same task. Of those, 240 are a *correct plan* for the same
  (start, target) pair, so the output is competent for a question that was not asked.
- 226/480 still carry the correct `RESULT:` prefix while containing a
  program, so this is a collapse of the answer *content*, not merely of the wrapper.

The registered `pass@1 = 0.00, parse rate = 0.00` therefore measures a collapse of
mode-conditional response behaviour, not destruction of modular arithmetic. The direct
cause is recorded in the drivers: `run_capacity_series.sh` and `run_amended_series.sh`
pass `--apply-weight 0`, and `train_sft.py` builds the stream as
`apply_tasks * apply_weight + plan_tasks`, so the 1875-step composition run saw zero
APPLY targets. The 20% replay declared in the capacity preregistration was PLAN-only.

This does not cancel the retention concern - a policy that cannot be addressed in APPLY
mode is not a usable multi-mode policy - but what these data license is *mode collapse
caused by removing APPLY from the training mixture*, not catastrophic forgetting of a
skill. The `applykeep` arm of the night series tests the causal claim directly.

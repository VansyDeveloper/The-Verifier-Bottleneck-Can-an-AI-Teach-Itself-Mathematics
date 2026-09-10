# Recovery Amendment 001 — pre-freeze confirm-atomic feasibility

## Trigger and blindness

After all six equal-budget training pairs had completed, freeze attempt 0 failed
before any final file was written. The exact error was
`confirm atomic generation exhausted for SH1, p=7, degree=3`. At the amendment
decision point, `CONFIG_FROZEN.json`, every `final_*.jsonl`,
`confirm_atomic.jsonl`, final manifests, final-access receipts, rankings,
metrics, and model evaluations were absent. Final A/B/C/D rows had existed only
transiently in process memory and were neither persisted, summarized, scored,
nor inspected.

The byte-identical failure receipt and stdout/stderr are retained under
`runs/failures/freeze_attempt0/` and are hash-bound by the machine-readable
amendment receipt.

## Root cause

The legacy atomic-reference corpus already occupies every nonzero candidate in
several requested field/degree cells. In particular, all 2,401 states for
`p=7, degree=3` are forbidden before pilot or final exclusions are added.
Increasing the rejection-sampling attempt limit or retrying another seed cannot
make that cell feasible.

## Frozen scientific scope

This amendment changes only the descriptive, explicitly non-gating
`confirm_atomic` generator. It does not change:

- Qwen3-0.6B or the frozen atomic export;
- any of the 12 trained adapters, training examples, optimizer settings, or
  budgets;
- final A/B/C/D generation, seeds, sizes, motifs, or disjointness rules;
- the final-A depth-3 exact Hit@32 primary endpoint;
- the success gates or the registered statistical analysis;
- the rule that final data may be evaluated only after `CONFIG_FROZEN.json`.

There is no retraining and no tuning on final data. Only the freeze phase is
retried once under this amendment.

## Deterministic repair

The original per-operation requested field×degree schedule and seed 73200 are
retained. Candidate states within a cell are scanned once in exhaustive
lexicographic order. If a requested cell contains no remaining exact,
state/task-disjoint candidate, the other registered cells are visited in a
single seed-derived order fixed by operation and row index. The first valid
candidate is used. No model score, generation, metric, alternate seed, or final
summary participates in fallback selection.

The repaired generator must produce exactly 200 rows for each of SH1, SC2, REV,
AC1, and AX1, retain exact verifier correctness, and pass the same unified
atomic/pilot/final state- and task-disjointness audit. Requested versus realized
cell counts, candidate scans, exhausted cells, and every fallback are written to
an audit receipt.

Before committing this amendment, one data-only regression execution recreated
the deterministic final splits and repaired confirm-atomic set in memory. It
produced the registered 4,000 composition rows/probes plus 1,000 atomic rows
(5,000 rows total), with 200 atomic rows per operation; the final, atomic, and
unified exact audits all passed. The atomic JSONL digest was
`DD084D13159D545F85F4FC89A3A47898D22A853302A2481D7301B45AC4E3373F`.
No row content, model output, ranking, metric, or alternative seed was used to
alter the algorithm after this check.

## Recovery chain

Before retry, a clean Git commit and machine-readable amendment receipt must
bind the original preregistration, selection freeze, budget manifest,
`TRAINING_DONE`, all six pair receipts, all 12 adapter trees and training
receipts, freeze-attempt-0 evidence, the no-final-access snapshot, and the new
generation code. The original preregistration, protocol, selection freeze,
budgets, training receipts, and `FAILED.json` are not overwritten.

Immediately before generation, the implementation atomically creates one
immutable `freeze_attempt1_STARTED.json` receipt. A process interruption may
resume that same deterministic attempt from hash-matching partial generation
files, but cannot create a second attempt; final access, rankings, metrics, or
analysis still make resume fail closed. `CONFIG_FROZEN.json` binds the STARTED
receipt, and a DONE receipt binds the completed freeze before evaluation opens.

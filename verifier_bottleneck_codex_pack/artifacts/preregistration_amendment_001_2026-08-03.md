# Amendment 001 to the D-021 capacity preregistration

Dated 2026-08-03. Amends `preregistration_capacity_2026-08-02.md`, which stays
frozen and is not edited. Nothing already registered is withdrawn; two endpoints
are added and one reporting error is corrected.

## Trigger

Independent external materials arrived after the D-021 preregistration was frozen
and before its data were opened (`new_stuff/stage4_verifier_bottleneck_audit_v22`).
They report a preregistered confirmation on the **same base model and the same
field split** — known fields 5/7/13/19/23/29, transfer fields 11/17 — using
exhaustive ranking rather than sampling. Their published 2x2:

| | known fields | transfer fields 11/17 |
|---|---:|---:|
| ordinary motifs | A: **+27.4 pp** | C: **+26.1 pp** |
| held motifs | B: **−16.0 pp** | D: **−11.8 pp** |

where a *motif* is an adjacent operation pair and the held set is
`AX1->SH1`, `AC1->REV`, `SC2->AX1`.

Two consequences for our design:

1. **Field novelty is nearly free (~1 pp); motif novelty costs ~40 pp and flips
   the sign.** Our registered 2x2 varies depth and field only, so it cannot
   observe the axis on which composition supervision actually fails. Our C1 was
   therefore likely to return a large positive and be read as "method ceiling"
   while the real ceiling sat on an axis we were not measuring.
2. **Their protocol forbids sampled pass@K and random generation**, ranking all
   5^d programs instead. That is the direct remedy for the confound this project
   already established: at low temperature the policy spends 32 draws on ~3
   distinct programs, so sampled `pass@32` measures sampler diversity as much as
   policy preference.

## Addition 1 — exhaustive ranking as the co-primary metric

`Hit@K` over the complete candidate set: all 5^d programs scored, ranked by
`sum_t log_softmax(action scores at prefix_t)[action_t]`, ties broken
lexicographically by operation signature. Temperature fixed at 1.0. No sampler,
no seed; the metric is deterministic given the adapter.

The registered sampled `pass@32` endpoints remain and will be reported. Where the
two disagree, exhaustive `Hit@K` is the primary evidence, because it is immune to
the duplicate-collapse confound by construction.

**Correction to the registered reporting plan.** Depth-2 tasks have only
5^2 = 25 candidate programs, so `Hit@32` is trivially 1.000 there and carries no
information. Depth-2 is reported at `Hit@1` and `Hit@8`; `Hit@32` is used only for
depth 3, where the candidate set is 125. This was an error in the original plan,
not a change of endpoint after seeing results — it is a property of the candidate
counts, fixed before the depth-2 numbers were interpreted.

## Addition 2 — the motif axis (D-022)

New training set `sft_train_composition_nomotif` (4000 programs, train fields,
depths 2 and 3) with every held-motif program removed, and four evaluation
families of 250 depth-3 tasks each:

| split | fields | held motif | sha256 |
|---|---|---|---|
| `motif_a_d3` | known | absent | `1b15e97223f7…` |
| `motif_b_d3` | known | required | `e453a04c6e02…` |
| `motif_c_d3` | 11, 17 | absent | `81204ed543d9…` |
| `motif_d_d3` | 11, 17 | required | `aab3fd895f46…` |

A task enters a held-motif family only if **every** minimal-depth correct program
contains a held motif; otherwise an unheld route would solve it and the split
would not test the motif. Non-motif families require the converse. All splits are
exact-key disjoint from each other and from every pre-existing split.

Adapters: three seeds trained on the no-motif set, against the shared atomic
checkpoint as baseline.

### New endpoints

- **M1 (primary for D-022)** — `Hit@32(nomotif) − Hit@32(atomic)` on `motif_b_d3`.
  Does composition supervision transfer to unseen operation pairs in known fields?
- **M2** — the same on `motif_d_d3` (unseen pairs and unseen fields together).
- **M3** — the same on `motif_a_d3` and `motif_c_d3`, the in-distribution controls.
- **M4** — the motif main effect: `(M3 mean) − (M1, M2 mean)`.

Decision rule for M1, fixed here: CI above zero means motif transfer succeeds and
the external negative does not replicate; CI containing zero means supervision
does not transfer to unseen motifs; CI below zero means supervision actively harms
unseen motifs, replicating the external B/D result.

## Unchanged

Model, field split, operation set, LoRA rank and module set, optimiser budget,
the shared atomic starting checkpoint, the D-021 primary arm choice (`iid@0.7`
for sampled endpoints), bootstrap procedure, and the joint C1/C2 interpretation
table. No arm, adapter, temperature or K may be re-selected after results are seen.

## Disclosure

The two D-021 adapters for seeds 1 and 2 failed on their first attempt: the
machine suspended mid-run and the CUDA context did not survive resume
(`CUBLAS_STATUS_EXECUTION_FAILED`, then `CUDA unknown error`). The failure
receipts are retained as FAILED run directories. The adapters were retrained from
the identical command and identical data; no configuration changed.

A cross-prefix batching optimisation for the ranker was implemented, measured to
perturb action log-probabilities by up to 0.2 nats under bf16 through batch-size
dependent GEMM kernels, and removed rather than adopted. Every prefix is scored in
its own fixed 5-row call. The measurement and the reason are recorded in
`scripts/run_exhaustive_ranking.py`.

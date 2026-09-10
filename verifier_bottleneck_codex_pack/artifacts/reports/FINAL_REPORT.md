# The Verifier Bottleneck — final report

2026-08-04. Master document for the exploration × composition axis. Supersedes
the interpretation in `final_report.md` (July) and consolidates
`final_report_2026-08-01.md` and the D-021/D-022 ranking report. Every number here
is traceable to raw run artifacts and the machine-readable statistics files.

---

## 1. The question, and the answer

> At a fixed compute budget, does structured exploration of the solution space
> help a model discover and consolidate new compositions of previously learned
> atomic skills, or does self-training merely redistribute probability over
> answers that were already reachable?

For this sandbox, **search and self-training do not expand reachability, although
direct supervision can teach the tested compositions**:

1. **Structured exploration does not help.** Prefix-balanced action search
   carries no advantage over iid sampling once the iid baseline is tempered to
   the same candidate diversity: **−1.11 pp, 95% CI [−4.22, +1.89]**, 0 of 3
   seeds. Uniform-random program sampling beats it by **+11.22 pp** and the best
   iid arm by **+12.67 pp**, both 3/3 seeds.
2. **Self-training does not expand reachability.** After the full 400-step
   protocol with an exact verifier and a training sampler tuned to explore
   properly, the change in held-out `pass@32` is **+0.22 pp, CI [−3.11, +3.67]**.
   The policy demonstrably learned — per-candidate accuracy rose 2.4× — but it
   lost exactly as much coverage as it gained in sharpness.
3. **The earlier nulls are not a hard PLAN-capacity ceiling.** Under exhaustive
   ranking, direct composition SFT improves depth-3 Hit@32 by **+48.22 pp** on
   known fields and **+59.11 pp** on fields 11/17.
4. **That gain is narrow and destructive.** The same SFT hurts held operation
   motifs by **−13.47 pp** and **−15.07 pp**, and all six composition adapters
   collapse on APPLY from pass@1 **1.00 to 0.00**. It is evidence of PLAN
   ranking capacity, not retained multi-mode competence.

The July pilot's headline of +16.0 pp is real and reproduces at **+16.33 pp**.
It was never a bug. It was a missing control.

---

## 2. What the July pilot actually measured

`docs/05` specifies exploration controls E0–E4. In the action space only E1 (iid,
T=0.7) and E4 (prefix-balanced) were ever run: the single screen covering E0–E4
(`20260727T094346Z`) was the free-text variant and returned 0.0 correct for all
five methods. Structured search was therefore only ever compared against an iid
baseline that was drastically under-tempered.

How under-tempered, measured on dev with the shared SFT checkpoint:

| iid temperature | dev pass@32 | distinct programs / task |
|---|---:|---:|
| 0.7 — the July control | 0.060 | 3.11 |
| 1.6 | 0.190 | 8.04 |
| 2.0 | 0.250 | 9.88 |
| 3.0 | 0.390 | 14.57 |
| 5.0 | 0.460 | 19.17 |
| **8.0 — optimum** | **0.500** | 21.92 |
| 100.0 — effectively uniform | 0.420 | 23.74 |
| temperature mix (E3) | 0.090 | 4.25 |
| **prefix-balanced @ 0.7** | **0.290** | 9.63 |

At T=0.7 the policy spends 32 candidates on 3.1 distinct programs. Prefix-balanced
reaches 9.6 by forcing five different first actions. That gap is the whole effect.

The interior optimum at T≈8 also settles a separate question: the model's action
scores are *not* noise. Uniform sampling has more diversity yet scores lower
(0.420 vs 0.500). The policy contributes real information — just not enough to put
prefix-balanced ahead of random search.

Two further defects, both visible in the July artifacts:

- The July frozen set was evaluated only on the diagonal — iid adapter with iid
  search, prefix adapter with prefix search — so the search effect and the
  training effect were never separated. The July dev runs show the same +12 pp
  gap on the **pre-GRPO** adapter, i.e. before any RL.
- `unique_correct_programs` was the raw correct-candidate count, not distinct
  programs. True values over the 200 July tasks: 13–14 (iid) and 26–33 (prefix),
  reported as 114/159, 88/171 and 141/154.

---

## 3. Design

The first two preregistrations were each frozen before their test set was opened,
with their decision rules fixed in advance.

| | D-018 | D-020 |
|---|---|---|
| question | structure vs diversity | self-training with adequate exploration |
| frozen set | `confirmatory_heldout.jsonl`, 300 tasks | `confirmatory_heldout_v2.jsonl`, 300 tasks |
| sha256 | `4a2e4fd1…96d45` | `e75320e0…422892` |
| preregistration | `preregistration_confirmatory_2026-08-01.md` | `preregistration_selftraining_2026-08-01.md` |
| selection on dev | T\*=2.0 (diversity-matched), T_best=3.0 | T_train=8.0 (dev optimum) |

Both sets are PLAN, depth exactly 3, held-out primes 11 and 17, degree caps 2 and
3, order-sensitivity required, and have zero exact-key collisions with each other
or with any other split.

Training in both series: GRPO, 400 steps (`docs/03` Phase G), group size 8,
lr 1e-5, KL beta 0.01, exact binary reward, seeds 0/1/2, every branch starting
from the same `sft_atomic_r32_pilot_aw4_cont` adapter, prompts from `rl_train`.
The only variable D-020 changes is the sampling temperature during training.

Evaluation: K=32, programs of exactly `max_steps` operations, exact verifier on
complete programs only. Statistics per `docs/07` 7.7 — paired per-task
differences, 10,000-fold hierarchical bootstrap, aggregation inside a seed then
across seeds, exact McNemar, Holm across each declared secondary family.

D-021 was separately frozen in `preregistration_capacity_2026-08-02.md` before
its four capacity sets were opened. Amendment 001, frozen before the D-021 data
were opened, added D-022's held-motif 2×2 and exhaustive ranking endpoints while
leaving the original preregistration intact.

---

## 4. Results

### 4.1 D-018 — structure versus diversity

`confirmatory_heldout`, 300 tasks, 3 seeds.

| contrast | delta | 95% CI | seeds | Holm p |
|---|---:|---|---:|---:|
| **Q1 primary** prefix − iid@2.0 | **−1.11 pp** | [−4.22, +1.89] | 0/3 | p=0.527 |
| Q2 GRPO effect, iid branch | −0.11 pp | [−0.89, +0.67] | 1/3 | 1.0 |
| Q2 GRPO effect, prefix branch | −1.89 pp | [−3.78, +0.00] | 0/3 | 0.142 |
| Q3 published diagonal | +9.00 pp | [+6.89, +11.11] | 3/3 | 1.0e-15 |
| Q4 iid@3.0 − prefix | +9.78 pp | [+6.44, +13.00] | 3/3 | 4.1e-08 |

Q1 falls in the preregistered "CI contains zero" branch: the prefix-balanced
advantage is not distinguishable from a diversity effect.

`pass@32` is a near-monotone function of realised candidate diversity, and
prefix-balanced sits *on* that curve. Plotted against cost rather than diversity
it sits clearly *below* it: 0.146 at 4248 scoring passes, against 0.243 for
`iid@3.0` at 4301 — see `confirmatory_diversity_curve.png`.

### 4.2 Continuity on the July set

`final_like_heldout`, 200 tasks, already opened in July, so re-analysis only.

| contrast | delta | 95% CI | seeds | Holm p |
|---|---:|---|---:|---:|
| Q3 published diagonal | **+16.33 pp** | [+13.17, +19.50] | 3/3 | 3.8e-25 |
| Q2 GRPO effect, iid branch | +0.17 pp | [−0.83, +1.17] | 1/3 | 1.0 |
| Q2 GRPO effect, prefix branch | −1.33 pp | [−4.17, +1.50] | 1/3 | 0.83 |

An independently rebuilt pipeline on different hardware reproduces the published
+16.0 pp at +16.33 pp. The July measurement was sound; its interpretation was not.

### 4.3 D-020 — self-training with adequate exploration

`confirmatory_heldout_v2`, 300 tasks, 3 seeds.

| contrast | delta | 95% CI | seeds | Holm p |
|---|---:|---|---:|---:|
| **Q5 primary** hot-GRPO − pre-GRPO, at iid@8.0 | **+0.22 pp** | [−3.11, +3.67] | 2/3 | p=0.949 |
| Q6 hot − cold training sampler | +0.89 pp | [−2.45, +4.22] | 2/3 | 0.647 |
| Q7 uniform random − prefix | **+11.22 pp** | [+7.33, +15.11] | 3/3 | 5.5e-08 |
| Q8 best iid − prefix | **+12.67 pp** | [+8.89, +16.56] | 3/3 | 7.7e-10 |

Q5 falls in the "CI contains zero" branch. The D-018 null was not an artefact of
a cold training sampler: raising the training temperature from 0.7 to 8.0, where
the sampler demonstrably explores, changes nothing on held-out data.

### 4.4 The mechanism

Measured at the `iid@0.7` arm, where the policy's own preferences dominate:

| adapter | per-candidate accuracy | distinct programs / task | pass@32 |
|---|---:|---:|---:|
| pre-GRPO | 0.0151 | 3.80 | 0.044 |
| GRPO T_train=0.7 | 0.0139 | 2.49 | 0.038 |
| **GRPO T_train=8.0** | **0.0358** | **1.81** | 0.061 |

The hot-trained policy is **2.4× more accurate per sample** — it genuinely learned
from the exact reward. It is also **2.1× narrower**. The two cancel, and `pass@32`
moves by 1.7 pp. This is probability redistribution rendered in three numbers
(`selftraining_mechanism.png`).

### 4.5 Best configuration in the entire project

`pass@32 = 0.331`, hot-GRPO adapter searched with uniform-random program sampling
— statistically indistinguishable from the pre-GRPO adapter with the same arm
(0.308) and from pre-GRPO with `iid@8.0` (0.322). No trained adapter is reliably
better than the SFT starting point at any arm.

### 4.6 Atomic forgetting after GRPO

Greedy pass@1 on frozen atomic validation, all trained adapters plus the start:

- non-SH1 APPLY: **1.00 (80/80) everywhere**;
- PLAN: 0.98 pre-GRPO, 0.97–0.98 after GRPO, worst case one percentage point.

No atomic collapse. The absence of held-out gains is not explained by damage.

### 4.7 D-021 — capacity control by exhaustive ranking

All 125 depth-3 programs are ranked exactly; the metric contains no sampling
seed or duplicate-candidate confound. The 12-task smoke run is rejected by exact
task-count matching.

| contrast | delta Hit@K | crossed seed × task 95% CI |
|---|---:|---:|
| **C1** composition SFT − atomic, fields 11/17, Hit@32 | **+59.11 pp** | [+53.00, +65.33] |
| **C2** composition SFT − atomic, known fields, Hit@32 | **+48.22 pp** | [+41.33, +55.00] |
| C3 depth-2 − depth-3, Hit@8 | +42.89 pp | [+38.00, +47.67] |
| C4 known − held-out fields, Hit@32 | −2.22 pp | [−6.00, +1.56] |

C1 and C2 select the preregistered **method-ceiling** branch for PLAN ranking:
the tested model and LoRA rank can represent these depth-3 programs, including
on new fields. This does not isolate search from policy-gradient credit
assignment, supervision density or optimisation budget.

The frozen retention rule changes the overall verdict. Every composition
adapter retains PLAN pass@1 at 0.91–0.98 but loses APPLY completely: pass@1
1.00 → 0.00 and parse rate 1.00 → 0.00. Therefore the capacity gain is not
counted as a clean success.

### 4.8 D-022 — held-motif transfer

| family | composition SFT − atomic, Hit@32 | crossed seed × task 95% CI |
|---|---:|---:|
| A: known fields, seen motifs | +55.20 pp | [+48.13, +62.00] |
| B: known fields, held motifs | **−13.47 pp** | [−24.00, −2.67] |
| C: fields 11/17, seen motifs | +58.80 pp | [+51.47, +65.73] |
| D: fields 11/17, held motifs | **−15.07 pp** | [−25.60, −4.27] |

The motif main effect is **+71.27 pp** [58.53, 83.60]; the field main effect is
**−1.00 pp** [−8.27, 6.33]. The sign flip reproduces the external qualitative
result on independently generated data. Because A/B/C/D are all PLAN tasks,
the interaction cannot be explained by the separate APPLY mode collapse.

### 4.9 Registered sampled D-021 grid (secondary)

The original K=32 sampled grid was completed for continuity. The registered
primary `iid@0.7` contrasts are:

| contrast | delta pass@32 | hierarchical 95% CI |
|---|---:|---:|
| **C1** composition SFT − atomic, fields 11/17 | **+63.33 pp** | [+56.67, +70.33] |
| **C2** composition SFT − atomic, known fields | **+62.56 pp** | [+56.44, +68.44] |
| C3 depth-2 − depth-3 | +26.44 pp | [+19.67, +33.33] |
| C4 known − held-out fields | +1.11 pp | [−5.33, +7.67] |

C1 and C2 again select the method-ceiling branch. This grid remains secondary:
random K=32 evaluation also measures sampler diversity, whereas exhaustive
ranking isolates policy ordering. Its measured cost was 235,197 model passes,
2,116,773 scored operation tokens and 6.62 GPU-hours.

---

## 5. Integrity and reproduction

### 5.1 The checkpoint chain was rebuilt from nothing

The July adapters did not survive: `artifacts/adapters/` is gitignored and was
absent, no run directory retained `checkpoints/`, and the delivered archive
contained no weight files. The SFT chain was rebuilt on different hardware under
D-017 and reproduces the D-013 gate exactly:

| metric | July | rebuilt |
|---|---:|---:|
| non-SH1 APPLY pass@1 | 1.00 (80/80) | **1.00 (80/80)** |
| PLAN pass@1 | 0.98 (98/100) | **0.98 (98/100)** |
| full APPLY pass@1, reported separately | 0.83 | **0.83** |

### 5.2 Memory-only changes, verified rather than asserted

GRPO peaked at 7.27 GB in July and had to fit a 4 GB card. Under D-017: one base
model carrying two copies of the adapter instead of two models; `logits_to_keep`
on the language-model head; per-position backward; token-weighted micro-batching
in SFT. Peaks became 2.36 GB (SFT), 3.13 GB (GRPO), 1.64 GB (evaluation).

Equivalence is pinned by tests, not by claim:
`tests/test_action_policy.py` compares the optimised scorer against a verbatim
copy of the original on a real forward pass (atol/rtol 1e-4); micro-batch
gradients match the full batch to 5.5e-6 relative.

### 5.3 Data integrity — what the audit found

`scripts/verify_data_integrity.py` checks all ten splits by exact key rather than
asserting cleanliness. It found leaks that July's Gate 3 tick had missed:
`sft_train_apply ^ sft_validation_apply` 12, `sft_train_plan ^ sft_validation_plan`
11, `sft_calibration_sh1 ^ sft_validation_apply` 5, `rl_train ^ dev_exploration` 1.

Impact, recomputed on never-trained items only:

| gate metric | all items | never-trained only |
|---|---:|---:|
| non-SH1 APPLY | 1.00 (80/80) | 1.00 (72/72) |
| PLAN | 0.98 (98/100) | 0.9775 (87/89) |
| full APPLY | 0.83 (83/100) | **0.8488 (73/86)** |

The leak did not inflate the gates — full APPLY is *higher* on unseen items,
because the overlapping examples were SH1-heavy. T\* selection is unchanged when
the one leaked dev task is dropped (9.630 → 9.646 distinct/task; T\* stays 2.0).
Recorded in D-019 with the deviation that atomic SFT used only primes 5 and 7
against the six declared in `docs/03` 3.3 — conservative in direction.

Gate-by-gate evidence: `artifacts/gate_checklist_2026-08-01.md`, including the
lines that were wrongly ticked in July.

---

## 6. Limitations

- One base model (Qwen3-0.6B-Base), one task family, one LoRA rank, one reward.
  The claim is about this sandbox, not about structured exploration in general.
- Temperature here divides a summed sequence log-probability, not a per-token
  logit. These values are not comparable to ordinary decoding temperatures.
- T\*, T_best and T_train were selected on 100 dev tasks at one seed. The
  decision rules were preregistered; the temperatures themselves are estimates.
- 300 tasks × 3 seeds resolves a contrast to roughly ±3 pp. Effects smaller than
  that would not have been detected — "not distinguishable from zero" is the
  correct reading of Q1 and Q5, not "exactly zero".
- Prefix-balanced forces first actions at candidates 0..K−2 by `candidate_id % 5`,
  giving SH1 one extra slot at K=32 (31 forced + 1 free rather than `docs/05`
  5.6's 30 + 2). Left unchanged for comparability with the published numbers.
- All D-018, D-020, capacity and motif evaluation sets are now spent. Any further
  confirmatory claim needs a newly generated set and a new preregistration.
- Direct composition SFT catastrophically forgets APPLY. It answers a narrow
  representational-capacity question and is not a usable multi-mode policy.
- Phase H, the verifier-quality axis, is not run and is **not interpretable as
  specified**: with the self-training effect indistinguishable from zero under a
  *perfect* verifier, degrading the verifier has no effect to degrade. It becomes
  meaningful only if some configuration first produces a non-null gain.

---

## 7. What is and is not licensed

Supported:

> In this finite-algebra composition sandbox, at a fixed candidate budget, the
> advantage of prefix-balanced action-space exploration over iid sampling is
> explained by the number of distinct programs the search actually tries. Once an
> iid baseline is tempered to the same diversity the advantage disappears, and
> both the best iid arm and uniform-random program sampling exceed it at equal
> or lower cost. Four hundred steps of GRPO with an exact verifier measurably
> sharpen the policy — per-candidate accuracy rises 2.4× — but narrow it by the
> same factor, producing no measurable expansion of effective reachability, with
> or without an adequately exploring training sampler. Direct composition
> supervision shows that depth-3 PLAN programs are rankable on known and new
> fields, but it harms unseen operation motifs and destroys APPLY behaviour.

Not supported: any claim that structured exploration creates or consolidates new
compositions here; any claim that this self-training loop expanded effective
reachability; any statement that the base policy has zero probability of a given
program. The negative primary results are retained in full, as `AGENTS.md`
requires.

A constructive corollary worth stating: in this sandbox the cheapest large gain
available was not an algorithm but a sampling temperature — `pass@32` moved from
0.038 to 0.322 by changing one scalar, roughly eight times the effect the July
series attributed to structured exploration.

---

## 8. Reproduction

```bash
cd verifier_bottleneck_codex_pack
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements/lock_linux_cu124.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preflight.py --output artifacts/environment/preflight_linux_3050ti.json
.venv/bin/python scripts/verify_data_integrity.py

# atomic checkpoint chain: D-009 then D-010
.venv/bin/python scripts/train_sft.py --config configs/training/sft_atomic.yaml \
  --tier pilot --seed 0 --max-steps 1875 --lora-config configs/training/lora_r32.yaml \
  --apply-weight 4 --micro-batch-size 2 --output-name sft_atomic_r32_pilot_aw4
.venv/bin/python scripts/train_sft.py --config configs/training/sft_atomic.yaml \
  --tier pilot --seed 0 --max-steps 625 --lora-config configs/training/lora_r32.yaml \
  --apply-weight 4 --learning-rate 5e-5 --micro-batch-size 2 \
  --base-adapter artifacts/adapters/sft_atomic_r32_pilot_aw4 \
  --output-name sft_atomic_r32_pilot_aw4_cont

# frozen sets
.venv/bin/python scripts/prepare_confirmatory_heldout.py
.venv/bin/python scripts/prepare_confirmatory_heldout.py --seed 20260802 \
  --output artifacts/data/pilot/confirmatory_heldout_v2.jsonl

# dev arm selection — fixes T*, T_best, T_train; never touches confirmatory data
.venv/bin/python scripts/run_action_search_screen.py \
  --adapter artifacts/adapters/sft_atomic_r32_pilot_aw4_cont \
  --input artifacts/data/pilot/dev_exploration.jsonl --k 32 --seed 0 \
  --task-tag plan-dev --run-method-tag diversity-screen \
  --methods iid_action@0.7 iid_action@1.0 iid_action@1.3 iid_action@1.6 \
            iid_action@2.0 iid_action@2.5 iid_action@3.0 \
            temp_mix_action@0.2,0.6,1.0,1.4 prefix_balanced_action@0.7
.venv/bin/python scripts/run_action_search_screen.py \
  --adapter artifacts/adapters/sft_atomic_r32_pilot_aw4_cont \
  --input artifacts/data/pilot/dev_exploration.jsonl --k 32 --seed 0 \
  --task-tag plan-dev --run-method-tag diversity-screen-ext \
  --methods iid_action@3.5 iid_action@4.0 iid_action@5.0 iid_action@6.0 \
            iid_action@8.0 iid_action@100.0

bash scripts/run_confirmatory_grpo.sh      # D-018 training, 6 runs x 400 steps
bash scripts/run_confirmatory_eval.sh      # D-018 evaluation, 18 cells
bash scripts/run_selftraining_series.sh    # D-020 training + evaluation
bash scripts/run_atomic_forgetting.sh

PYTHONPATH=scripts .venv/bin/python scripts/analyze_confirmatory.py \
  --dataset confirmatory_heldout --output artifacts/reports/confirmatory_statistics.json
PYTHONPATH=scripts .venv/bin/python scripts/analyze_confirmatory.py \
  --dataset final_like_heldout --contrasts Q2iid Q2prefix Q3 \
  --output artifacts/reports/july_reanalysis_statistics.json
PYTHONPATH=scripts .venv/bin/python scripts/analyze_selftraining.py \
  --output artifacts/reports/selftraining_statistics.json
.venv/bin/python scripts/plot_confirmatory_2026_08_01.py
.venv/bin/python scripts/plot_selftraining.py
.venv/bin/python scripts/analyze_results.py --runs artifacts/runs --output artifacts/reports

# D-021/D-022 controls
bash scripts/run_capacity_series.sh
.venv/bin/python scripts/analyze_capacity.py
.venv/bin/python scripts/analyze_ranking.py
.venv/bin/python scripts/plot_ranking_results.py
```

Hardware: RTX 3050 Ti Laptop, 4 GB. Nine 400-step GRPO runs at 2355–2930 s each;
27 evaluation cells. Total measured GPU time for the two series ≈ 13 hours.

---

## 9. Artifact index

**Preregistrations** (frozen, not edited after freezing)
- `artifacts/preregistration_confirmatory_2026-08-01.md` — D-018
- `artifacts/preregistration_selftraining_2026-08-01.md` — D-020
- `artifacts/preregistration_capacity_2026-08-02.md` — D-021
- `artifacts/preregistration_amendment_001_2026-08-03.md` — D-022 and exhaustive ranking

**Statistics**
- `artifacts/reports/confirmatory_statistics.json` — Q1–Q4, arm table
- `artifacts/reports/july_reanalysis_statistics.json` — continuity
- `artifacts/reports/selftraining_statistics.json` — Q5–Q8, arm table
- `artifacts/reports/data_integrity.json` — all ten splits, leaks quantified
- `artifacts/reports/run_index.csv` — every run, including FAILED
- `artifacts/reports/capacity_statistics.json` — registered sampled D-021 grid
- `artifacts/reports/ranking_statistics.json` — D-021/D-022 exhaustive ranking and retention

**Figures**
- `confirmatory_diversity_curve.png` — structure on the diversity curve, and below it on cost
- `dev_temperature_sweep.png` — the interior optimum at T≈8
- `selftraining_mechanism.png` — accuracy up, coverage down, net zero
- `selftraining_arms.png`, `confirmatory_arm_by_role.png`
- `confirmatory_contrasts.png`, `selftraining_contrasts.png`
- `confirmatory_grpo_training.png`
- `ranking_contrasts.pdf`, `ranking_contrasts.png` — D-021/D-022 contrasts

**Decisions and gates**
- `DECISIONS.md` D-017 (memory-only changes with equivalence tests), D-018
  (structure vs diversity), D-019 (integrity audit), D-020 (self-training),
  D-021 (capacity control), D-022 (held motifs and exhaustive ranking)
- `artifacts/gate_checklist_2026-08-01.md` — all ten gates with evidence
- `artifacts/BLOCKERS.md`

**Superseded but preserved**
- `artifacts/reports/final_report.md` — July; data valid, interpretation replaced
- `artifacts/reports/final_report_2026-08-01.md` — D-018 only

# Structure or diversity? A preregistered refutation of the July pilot result

2026-08-01. Supersedes the interpretation in `final_report.md`, not its data.

## Summary

The July three-seed pilot reported **+16.0 pp** `delta pass@32` for prefix-balanced
action-space exploration over iid action sampling, and read it as evidence that
structured exploration expands effective reachability after self-training.

This series rebuilt the entire pipeline from scratch, reproduced that number
almost exactly (**+16.33 pp** on the same data), and then showed that it does not
mean what it was taken to mean:

1. **The effect is not structure.** Against an iid arm whose sampling temperature
   was tuned on dev to produce the same number of distinct candidate programs,
   prefix-balanced shows **−1.11 pp, 95% CI [−4.22, +1.89]**, positive in **0 of 3**
   seeds. The preregistered decision rule puts this in the "not distinguishable
   from a diversity effect" branch.
2. **Plain iid at a higher temperature is strictly better.** `iid@3.0` beats
   prefix-balanced by **+9.78 pp, CI [+6.44, +13.00]**, 3/3 seeds, at *identical*
   model-pass cost (4301 vs 4248 scoring passes over 300 tasks).
3. **The self-training stage contributes nothing.** After the full 400-step
   protocol, GRPO changes held-out `pass@32` by **−0.11 pp** (iid branch) and
   **−1.89 pp** (prefix branch); neither CI excludes zero. Every trained adapter
   is at best equal to, and usually slightly worse than, the shared SFT starting
   point on every search arm.

The best configuration in the entire series is the **pre-GRPO checkpoint searched
with plain iid at T=3.0**, `pass@32 = 0.243`.

## Why the July reading was wrong

`docs/05` specifies exploration controls E0–E4. In the action space only E1
(iid, T=0.7) and E4 (prefix-balanced) were ever run: the single E0–E4 screen
(`20260727T094346Z`) was the free-text variant and returned 0.0 correct for all
five methods. So "structured search" was only ever compared against an iid
baseline that was under-tempered.

How under-tempered: at T=0.7 the learned action policy is nearly deterministic
and spends 32 candidates on **3.7 distinct programs**. Prefix-balanced, by forcing
five different first actions, reaches **10.7**. Almost the whole gap is that.

Two further defects in the original design, both visible in its own artifacts:

- The July frozen set was evaluated only on the diagonal — iid adapter with iid
  search, prefix adapter with prefix search — so the search effect and the
  training effect were never separated. Re-analysis of the July dev runs shows the
  same +12 pp gap on the **pre-GRPO** adapter, i.e. before any RL.
- `unique_correct_programs` in the July metrics was the raw correct-candidate
  count, not a count of distinct programs. True unique correct programs over the
  200 July held-out tasks were 13–14 (iid) and 26–33 (prefix), reported as
  114/159, 88/171 and 141/154.

## Design

Preregistration frozen before any confirmatory data was opened:
`artifacts/preregistration_confirmatory_2026-08-01.md`.

- **Test set** `artifacts/data/pilot/confirmatory_heldout.jsonl` — 300 PLAN tasks,
  depth exactly 3, held-out primes 11 and 17, order-sensitivity required,
  sha256 `4a2e4fd1c7d1e9766ddc23c43014713beb8ddb2f5dc30a518c209137dca96d45`,
  zero exact-key collisions with any existing split. Never opened before freezing.
  The July set was re-run separately for continuity and is reported as
  re-analysis only.
- **Arms**, all at K=32 with programs of exactly `max_steps` operations:
  A1 `iid@0.7` (July control), A2 `iid@2.0`, A3 `iid@3.0`,
  A4 `prefix_balanced@0.7`, A5 temperature mix `0.2,0.6,1.0,1.4` (E3).
- **Arm selection on dev only.** T\* = 2.0 minimises the gap in distinct
  programs per task against prefix-balanced (9.88 vs 9.63); T_best = 3.0 has the
  highest dev `pass@32`. Run `20260801T080046Z`.
- **Training** per `docs/03` Phase G and `docs/06` 6.5: 400 steps, group size 8,
  lr 1e-5, sampling T=0.7, KL beta 0.01, exact binary reward, seeds 0/1/2, both
  branches from the same `sft_atomic_r32_pilot_aw4_cont` adapter.
- **Evaluation grid**: 3 adapters x 5 arms x 3 seeds on the confirmatory set,
  3 adapters x 2 arms x 3 seeds on the July set. 18 cells, all DONE.
- **Statistics** per `docs/07` 7.7: paired per-task differences, 10,000-fold
  bootstrap, aggregation inside a seed then across seeds, exact McNemar, Holm
  across the secondary family.

## Results

### Search arms, confirmatory set, mean of 3 seeds

| adapter | arm | pass@32 | distinct/task | unique correct | coverage | model passes |
|---|---|---:|---:|---:|---:|---:|
| pre-GRPO | iid@0.7 | 0.038 | 3.71 | 5.3 | 0.022 | 1373 |
| pre-GRPO | temp mix | 0.051 | 5.25 | 8.3 | 0.032 | 1730 |
| pre-GRPO | iid@2.0 | 0.157 | 12.03 | 27.7 | 0.115 | 3089 |
| pre-GRPO | **iid@3.0** | **0.243** | 17.47 | 42.3 | 0.186 | 4301 |
| pre-GRPO | prefix-balanced | 0.146 | 10.69 | 21.7 | 0.102 | 4248 |
| GRPO iid | iid@0.7 | 0.037 | 2.32 | 5.0 | 0.023 | 1099 |
| GRPO iid | iid@2.0 | 0.140 | 9.52 | 24.3 | 0.099 | 2757 |
| GRPO iid | iid@3.0 | 0.240 | 15.63 | 42.0 | 0.183 | 4178 |
| GRPO iid | prefix-balanced | 0.136 | 8.99 | 20.7 | 0.102 | 3871 |
| GRPO iid | temp mix | 0.051 | 3.41 | 8.3 | 0.033 | 1344 |
| GRPO prefix | iid@0.7 | 0.031 | 2.33 | 4.7 | 0.020 | 1085 |
| GRPO prefix | iid@2.0 | 0.144 | 8.81 | 24.3 | 0.104 | 2592 |
| GRPO prefix | iid@3.0 | 0.218 | 14.73 | 37.3 | 0.170 | 3997 |
| GRPO prefix | prefix-balanced | 0.127 | 8.94 | 20.0 | 0.098 | 3889 |
| GRPO prefix | temp mix | 0.040 | 3.34 | 6.3 | 0.026 | 1306 |

`pass@32` is a near-monotone function of realised candidate diversity, and
prefix-balanced sits on that curve rather than above it
(`confirmatory_diversity_curve.png`, left panel). Plotted against cost instead of
diversity it sits clearly *below* the iid curve (right panel).

### Preregistered contrasts, confirmatory set

| contrast | delta | 95% CI | seeds | McNemar p | Holm p |
|---|---:|---|---:|---:|---:|
| **Q1 primary** prefix − iid@2.0 | **−1.11 pp** | [−4.22, +1.89] | 0/3 | 0.527 | — |
| Q2 GRPO effect, iid branch | −0.11 pp | [−0.89, +0.67] | 1/3 | 1.0 | 1.0 |
| Q2 GRPO effect, prefix branch | −1.89 pp | [−3.78, +0.00] | 0/3 | 0.071 | 0.142 |
| Q3 published diagonal | +9.00 pp | [+6.89, +11.11] | 3/3 | 2.6e-16 | 1.0e-15 |
| Q4 iid@3.0 − prefix | **+9.78 pp** | [+6.44, +13.00] | 3/3 | 1.4e-08 | 4.1e-08 |

### Continuity on the July set (re-analysis, already opened)

| contrast | delta | 95% CI | seeds | Holm p |
|---|---:|---|---:|---:|
| Q3 published diagonal | **+16.33 pp** | [+13.17, +19.50] | 3/3 | 3.8e-25 |
| Q2 GRPO effect, iid branch | +0.17 pp | [−0.83, +1.17] | 1/3 | 1.0 |
| Q2 GRPO effect, prefix branch | −1.33 pp | [−4.17, +1.50] | 1/3 | 0.83 |

The July headline of +16.0 pp is reproduced at +16.33 pp by an independently
rebuilt pipeline on different hardware. The finding was never a software defect
or a fluke — it was a missing control.

### GRPO narrowed the policy

Distinct programs per task under `iid@0.7` fell from **3.71** (pre-GRPO) to
**2.32** (iid branch) and **2.33** (prefix branch). The same contraction appears
on every arm. 400 steps of GRPO with an exact binary reward sharpened the action
policy without expanding what it can reach — which is the "probability
redistribution" horn of the project's original research question, not the
"new capability" horn.

Training behaviour matched the July pattern: candidate-level mean reward was
indistinguishable between branches (0.044–0.054), while groups containing at
least one correct program differed roughly fourfold (iid 25/27/27 of 400,
prefix 108/96/96). That gap is again coverage, and it did not convert into
held-out gains.

### Atomic forgetting

Greedy pass@1 on the frozen atomic validation splits, all six trained adapters
plus the starting point:

- non-SH1 APPLY: **1.00 (80/80) everywhere**, including pre-GRPO;
- PLAN: 0.98 pre-GRPO; 0.97–0.98 after GRPO, worst case one percentage point.

No atomic collapse. The absence of held-out gains is not explained by damage to
the atomic skills.

## Reproduction of the checkpoint chain

The July adapters did not survive (`artifacts/adapters/` is gitignored and was
absent; no run directory retained `checkpoints/`; the delivered archive contained
no weight files). The SFT chain was rebuilt from scratch under D-017 on different
hardware and reproduces the D-013 gate exactly:

| metric | July | rebuilt |
|---|---:|---:|
| non-SH1 APPLY pass@1 | 1.00 (80/80) | **1.00 (80/80)** |
| PLAN pass@1 | 0.98 (98/100) | **0.98 (98/100)** |
| full APPLY pass@1, reported separately | 0.83 | **0.83** |

## Limitations

- One base model (Qwen3-0.6B-Base), one task family, one adapter rank. The claim
  is about this sandbox, not about structured exploration in general.
- T\* and T_best were selected on 100 dev tasks at a single seed. A different dev
  set could move them; the decision rule, not the temperature, was preregistered.
- The temperature sweep stopped at T=3.0, which was the best arm. The optimum may
  lie higher; no arm above 3.0 was run, so `iid@3.0` is a lower bound on what
  plain iid can do.
- Temperature here divides a summed sequence log-probability, not a per-token
  logit, so these temperature values are not comparable to ordinary text
  decoding temperatures.
- Prefix-balanced forces first actions at candidates 0..K−2 by `candidate_id % 5`,
  giving SH1 one extra slot at K=32 (31 forced + 1 free rather than the 30 + 2 of
  `docs/05` 5.6). Kept unchanged for comparability; recorded in D-018.
- `confirmatory_heldout` was opened once, by this series. It is now spent.
- Statistical power: 300 tasks x 3 seeds resolves Q1 to about ±3 pp. An effect of
  structure smaller than that would not have been detected.

## What this does and does not license

Supported:

> At a fixed candidate budget, the advantage of prefix-balanced action-space
> exploration over iid sampling in this sandbox is explained by the number of
> distinct programs the search actually tries. Once an iid baseline is tempered
> to the same diversity, the advantage disappears; a hotter iid baseline exceeds
> it at equal model-pass cost. 400 steps of GRPO with an exact verifier produced
> no measurable gain in effective reachability and narrowed the policy.

Not supported: any claim that structured exploration creates or consolidates new
compositions here, and any claim that this self-training loop expanded effective
reachability. The negative Q1 is retained in full, as required by `AGENTS.md`.

## Exact reproduction

```bash
cd verifier_bottleneck_codex_pack
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements/lock_linux_cu124.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preflight.py --output artifacts/environment/preflight_linux_3050ti.json

# atomic checkpoint chain, D-009 then D-010
.venv/bin/python scripts/train_sft.py --config configs/training/sft_atomic.yaml \
  --tier pilot --seed 0 --max-steps 1875 --lora-config configs/training/lora_r32.yaml \
  --apply-weight 4 --micro-batch-size 2 --output-name sft_atomic_r32_pilot_aw4
.venv/bin/python scripts/train_sft.py --config configs/training/sft_atomic.yaml \
  --tier pilot --seed 0 --max-steps 625 --lora-config configs/training/lora_r32.yaml \
  --apply-weight 4 --learning-rate 5e-5 --micro-batch-size 2 \
  --base-adapter artifacts/adapters/sft_atomic_r32_pilot_aw4 \
  --output-name sft_atomic_r32_pilot_aw4_cont

# frozen confirmatory set
.venv/bin/python scripts/prepare_confirmatory_heldout.py

# dev arm selection (fixes T* and T_best; never touches confirmatory data)
.venv/bin/python scripts/run_action_search_screen.py \
  --adapter artifacts/adapters/sft_atomic_r32_pilot_aw4_cont \
  --input artifacts/data/pilot/dev_exploration.jsonl --k 32 --seed 0 \
  --task-tag plan-dev --run-method-tag diversity-screen \
  --methods iid_action@0.7 iid_action@1.0 iid_action@1.3 iid_action@1.6 \
            iid_action@2.0 iid_action@2.5 iid_action@3.0 \
            temp_mix_action@0.2,0.6,1.0,1.4 prefix_balanced_action@0.7

bash scripts/run_confirmatory_grpo.sh        # 6 runs x 400 steps, ~4 h
bash scripts/run_confirmatory_eval.sh        # 18 evaluation cells, ~4 h
bash scripts/run_atomic_forgetting.sh

.venv/bin/python scripts/analyze_confirmatory.py --dataset confirmatory_heldout \
  --output artifacts/reports/confirmatory_statistics.json
.venv/bin/python scripts/analyze_confirmatory.py --dataset final_like_heldout \
  --contrasts Q2iid Q2prefix Q3 \
  --output artifacts/reports/july_reanalysis_statistics.json
.venv/bin/python scripts/plot_confirmatory_2026_08_01.py
.venv/bin/python scripts/analyze_results.py --runs artifacts/runs --output artifacts/reports
```

Hardware: RTX 3050 Ti Laptop, 4 GB. Peak allocated CUDA memory 2.36 GB (SFT),
3.13 GB (GRPO), 1.64 GB (evaluation). Six GRPO runs took 2355–2388 s each.

## Artifacts

- `artifacts/preregistration_confirmatory_2026-08-01.md` — frozen design
- `artifacts/reports/confirmatory_statistics.json` — all contrasts and arm table
- `artifacts/reports/july_reanalysis_statistics.json` — continuity
- `artifacts/reports/confirmatory_diversity_curve.png` — the central figure
- `artifacts/reports/confirmatory_arm_by_role.png`
- `artifacts/reports/confirmatory_contrasts.png`
- `artifacts/reports/confirmatory_grpo_training.png`
- `artifacts/reports/run_index.csv` — all 107 runs, including 4 FAILED
- `DECISIONS.md` D-017 (memory-only changes, with equivalence tests) and D-018
  (this design)

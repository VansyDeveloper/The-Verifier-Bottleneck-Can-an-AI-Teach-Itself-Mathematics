# Paired entropy ablation, frozen before outcome evaluation

## Question and scope

Does a fixed operation-choice entropy bonus change exact Hit@32 after IID
group-normalized categorical policy-gradient training? The implementation has
no clipping and is not labelled clipped GRPO. This is an exploratory paired
study with three independent training seeds. A task or generated answer is not
an independent inferential replicate.

## Frozen inputs

- Source model: `Qwen/Qwen3-0.6B-Base`, immutable Hub revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- Local `model.safetensors` SHA-256:
  `CD2A512003E2F9F3CD3C32A9C3573F820BB28C940F73C57B1DDAA983D9223EBA`.
- Both arms begin at the same D-010 atomic LoRA adapter
  `sft_atomic_r32_pilot_aw4_cont`. Its deterministic tree SHA-256 is
  `C47403B7F6474C4C1261572DA0253CDBCE576AE22AADCFE86DD64B20254B3E04`.
- IID training file `artifacts/data/pilot/rl_train.jsonl` has 500 tasks and
  SHA-256 `140E95B8192D0E19E79716385FCDD2361483B07990324F7F933C007AACE7715F`.
- A new depth-three held-out file has 1,000 tasks at primes 11 and 17, and
  SHA-256 `703049FB3B5655AFED95B2BADA36B3C85ADC2712EF5C03A8DBF54A12AFA9C57D`.
  It was generated with seed 20260923 before training. Its semantic tasks do
  not overlap the original pilot, true-composition, or earlier trajectory
  diversity task files. The same file is used for every arm and seed.
- The six saved original 400-step logs remain a separate observational audit.
  Archive SHA-256 is
  `6A028AF242859BFFB8FAB89DCAA4B88C063C39FD43BDDA327AF8540036B2FE61E`.

## Intervention

Seeds 0, 1, and 2 each form a pair. Both arms in one pair train and evaluate
on the same physical university RTX 2080 Ti with FP32 model computation,
as documented in `AMENDMENT_003.md`.
The trainable model uses non-reentrant activation checkpointing to fit the
assigned GPU, as documented in `AMENDMENT_004.md`.
H200 and the local RTX 5060 Ti are excluded. The control has entropy
coefficient 0. The intervention adds `0.01 * H(p)/ln(5)` at each unforced
operation-choice state, averaged over choices within the candidate loss.
The base objective, exact binary reward, group standardization, KL term,
initial adapter, task order, group size 8, temperature 0.7, learning rate
1e-5, and KL coefficient 0.01 are identical. The control is retrained with
the same new code. Training duration is fixed to 400 steps if a short
technical benchmark plus a 25 percent time margin supports completion of
all six trainings and evaluations by 08:00 MSK 24 September. Otherwise it
is 150 steps for both arms. The benchmark is not part of the analysis.
The initial FP16 timing choice in `DURATION_FREEZE.json` is superseded by
`AMENDMENT_003.md`. The successful 30-step FP32 technical benchmarks and a
10-task training-set ranking speed probe fixed a common duration of 150 steps
before any full pair or held-out evaluation. `DURATION_FREEZE_FP32.json`
records the calculation. The conservative 400-step estimate exceeds the
08:00 deadline after the required 25 percent margin.

## Endpoints and audit

The primary effect is mean paired difference in Hit@32, entropy bonus minus
control, over three training seeds. Each of 1,000 common tasks is ranked over
all 125 length-three programs by the original categorical action-policy
score at temperature 0.7. Ranking ties are broken lexicographically. Save
all program scores, correctness flags, task metrics, shard hashes, adapters,
logs, and failure evidence. Secondary endpoints are Hit@K for K=1..125,
total probability mass of correct programs, Shannon entropy over the 125
programs, entropy conditional on correctness, sampled group state fractions,
and distinct correct programs. These are exploratory.

For the primary paired difference, report each seed, mean, sample SD,
95 percent t interval, and exact two-sided sign-flip p value. With three
pairs, the smallest attainable two-sided sign-flip p value is 0.25. Keep
incomplete or failed pairs outside the aggregate. Do not pool this result
with the independent V100 withheld-pair study.

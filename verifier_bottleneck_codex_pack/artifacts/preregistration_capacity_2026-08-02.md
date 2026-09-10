# Preregistration — capacity ceiling and the depth/prime decomposition

Frozen 2026-08-02, before any evaluation on the four `capacity_*` sets.
Not edited after freezing.

## 1. Why

D-018 and D-020 both returned null primary results: structured exploration is not
distinguishable from candidate diversity, and 400 steps of GRPO with an exact
verifier do not expand effective reachability. Neither series can say *why*.

Two readings remain open, and they call for opposite next steps:

- **capacity ceiling** — the model cannot represent depth-3 compositions on
  unseen primes at all, so every method was measuring the same floor;
- **method ceiling** — the model can represent them, but neither search nor
  policy-gradient training finds them.

Nothing in the project so far distinguishes these, because the model was never
shown a correct composition. `docs/03` Phase D and D-002 deliberately restrict
SFT to depth 1. This series lifts that restriction once, as a declared control.

A second confound runs through every held-out set used so far: they change
**depth and prime at the same time** (depth 3 *and* primes 11/17). The 2x2 grid
below separates them.

## 2. Data

Built by `scripts/prepare_capacity_data.py`, all rejected against the exact keys
of every pre-existing split and of each other.

| split | n | primes | depth | sha256 |
|---|---:|---|---:|---|
| `sft_train_composition` | 4000 | 5,7,13,19,23,29 | 2 and 3 | `1a157c2c248b…` |
| `capacity_d2_train` | 300 | 5,7,13,19,23,29 | 2 | `86fbd22f717d…` |
| `capacity_d3_train` | 300 | 5,7,13,19,23,29 | 3 | `bacf9e3d08c7…` |
| `capacity_d2_heldout` | 300 | 11,17 | 2 | `7c66ffdd39e3…` |
| `capacity_d3_heldout` | 300 | 11,17 | 3 | `7657b19f7957…` |

Held-out primes appear in no training file. Full hashes in
`artifacts/data/pilot/capacity_manifest.json`.

## 3. Training

From the same shared checkpoint `sft_atomic_r32_pilot_aw4_cont` as every other
branch in this project:

- 4000 known-correct composition programs (depth 2 and 3, train primes) plus
  1000 atomic depth-1 PLAN examples as replay — 5000 total;
- three epochs over the fixed stream, 1875 steps, exactly mirroring D-009;
- learning rate 2e-4, LoRA r=32, unchanged module set;
- seeds 0, 1, 2, giving three independent adapters
  `sft_composition_oracle_seed{0,1,2}`;
- no held-out prime and no `capacity_*` evaluation task enters training.

This is supervision on the full correct sequence, i.e. the strongest signal the
sandbox can give short of showing the answer at test time. It is an upper bound
on what this model and rank can do, not a proposed method.

## 4. Evaluation

Arms at K=32: `iid@0.7`, `iid@3.0`, `iid@8.0`. Adapters: the atomic baseline
`sft_atomic_r32_pilot_aw4_cont` and the three composition adapters. Seeds 0/1/2,
each adapter evaluated at its matching seed. All four `capacity_*` sets.

The primary arm is **`iid@0.7`**, not the hot arm used in D-018/D-020. Capacity
is a question about what the policy believes; at T=8 the action distribution is
nearly flattened and would mask exactly the effect being measured. The same
contrast at `iid@8.0` is reported as a declared secondary.

## 5. Hypotheses and decision rules

Statistics as before: paired per-task differences, 10,000-fold hierarchical
bootstrap, aggregation inside a seed then across seeds, exact McNemar, 3 seeds.

### C1 — primary: does composition supervision transfer to unseen primes?

`pass@32(composition adapter) − pass@32(atomic adapter)` at `iid@0.7` on
`capacity_d3_heldout`.

### C2 — is the model able to learn compositions at all?

The same contrast on `capacity_d3_train`.

### C3 — depth effect, isolated

`pass@32(capacity_d2_train) − pass@32(capacity_d3_train)`, composition adapter.
Primes held constant.

### C4 — prime effect, isolated

`pass@32(capacity_d3_train) − pass@32(capacity_d3_heldout)`, composition adapter.
Depth held constant.

Holm across {C2, C3, C4}. C1 uncorrected.

### Joint interpretation, fixed in advance

| C2 | C1 | conclusion |
|---|---|---|
| CI contains 0 | — | **capacity ceiling.** The model cannot learn depth-3 composition even from direct supervision. Every earlier null, including D-018 Q1 and D-020 Q5, is a statement about the model, not about exploration or self-training. |
| CI above 0 | CI contains 0 | **generalisation ceiling.** Compositions are learnable in-domain but do not transfer across primes. The earlier nulls are about transfer, and the sandbox needs a harder-to-memorise split before any exploration claim is meaningful. |
| CI above 0 | CI above 0 | **method ceiling.** Capacity and transfer are both fine, so the earlier nulls belong to search and policy-gradient training. First-step credit assignment then becomes worth testing. |

C3 and C4 attribute the difficulty: if C4 is large and C3 small, the sandbox is
mostly testing prime transfer rather than composition depth, which would be a
design finding about the benchmark itself.

## 6. Atomic forgetting

Reported for every composition adapter on the frozen atomic validation splits,
same protocol as before. A large composition gain bought with atomic collapse is
not counted as success.

## 7. What would make this negative

A null C2 is the strongest possible negative for the project: it would mean the
task family is beyond this model at this rank, and that no exploration or
self-training result obtained in it can be interpreted as evidence about
exploration or self-training. That answer is reported in full if it occurs. No
arm, adapter, temperature or K may be re-selected after seeing these results.

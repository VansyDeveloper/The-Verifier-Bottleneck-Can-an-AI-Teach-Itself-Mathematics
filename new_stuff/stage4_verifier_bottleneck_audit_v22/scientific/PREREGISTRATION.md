# Stage 4 composition-only confirmation on Qwen3-0.6B

## Scientific question

Does equal-budget `composition_distill` improve exact ranking of unseen compositional programs over `atomic_control` when both branches start from the same frozen Qwen3-0.6B atomic export?

The historical exploratory pilot is immutable. It remains formally negative under its old atomic-forgetting and base-range gates. Its family-A result (`48.4% -> 71.8% Hit@32`, `+23.4 pp`) is used only to freeze this separate follow-up configuration. No pilot task is part of the new final test.

## Primary endpoint and success criterion

The primary endpoint is `composition_distill - atomic_control` on frozen final-A, depth 3, exact `Hit@32`. Every task ranks all 125 unique programs. Equal scores are broken lexicographically by operation signature. Random generation, sampled pass@K and post-hoc deduplication are forbidden.

A positive composition confirmation requires all of:

- mean seed delta at least 5 percentage points;
- lower bound of the two-sided 95% Student t interval over six seed deltas strictly above zero;
- two-sided one-sample seed t-test `p < 0.05`;
- positive delta in at least five of six independent confirmation replicates;
- complete raw rankings, zero leakage, the same frozen tasks for both branches and all replicates, and equal registered training budgets.

SH1, PLAN, APPLY and atomic forgetting are measured and published but are explicitly non-gating. Correct-program probability mass and the exact sign-flip and crossed seed-by-task bootstrap are sensitivity/descriptive analyses, not success gates.

## Independence and freeze order

The six confirmation replicates use new RNG streams 73000-73005. The pilot seed-0 adapters are not reused. Training data, optimizer configuration, final-data generator seeds, primary endpoint and analysis rules are committed and preregistered before final data exist. `CONFIG_SELECTION_FROZEN.json` is written before final generation. The evaluator cannot open final data until `CONFIG_FROZEN.json` binds every test SHA-256.

The exact tokenizer payload and default regex behavior used by the pilot are preserved. A generic Transformers warning about `fix_mistral_regex` is disclosed but the flag is not changed post-pilot, because changing tokenization would break the registered pilot-to-confirmation comparison. The tokenizer payload is covered by the frozen atomic-export hash.

## Scope of claims

A positive primary result supports improvement on unseen state/task-disjoint family-A compositions in known fields. Held-motif B, new-field C/D and depth probes are separately reported secondary endpoints. The historical negative dev-B result is disclosed; primary A success must not be described as proof of held-motif transfer.

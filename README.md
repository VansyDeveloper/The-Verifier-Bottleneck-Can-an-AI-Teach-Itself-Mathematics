# The Verifier Bottleneck

A compact review package for the SMILES 2026 AI-for-Mathematics project.

Start with the seven-slide [kickoff deck (PDF)](presentation/verifier-bottleneck-kickoff.pdf).
The editable [PowerPoint version](presentation/verifier-bottleneck-kickoff.pptx)
is available to everyone. This README is the single canonical experiment
contract; the rest of the former planning material was deliberately removed.
The project is peer-owned: contributors decide and review on equal footing.

## The Question

Can filtering a model's self-generated work create more held-out success at a
fixed sampling budget, rather than only making already-reachable answers easier
to sample?

We test this in a small, auditable setting. It is **not** a claim of a universal
law, true zero-support expansion, or a verdict on frontier-model RL.

The only experimental dials are the verifier signal under matched selection
volume and the candidate budget `K`; all other training and evaluation choices
stay fixed.

## Minimal Task

The sandbox composes affine maps over `F_101`:

```text
g(x) = a*x + b (mod 101), where a != 0
g3(g1(g4(x))) -> canonical answer (A, B)
```

The full map `(A, B)` has about 10,100 possibilities, is order-sensitive, and
can be checked exactly. The working split is:

- length-1 maps for warm-up;
- fresh length-2 compositions for the candidate pool and primary test;
- length-3 transfer only if the core result earns it.

The exact task manifest, answer grammar, exclusions, and split are frozen
before the first dry run.

The source-pool and held-out manifests are disjoint before any generation: no
held-out prompt, canonical answer, or ordered map tuple may enter sampling or
training. Held-out tasks must test unseen ordered compositions, not a random
reordering of source rows.

## Experiment Contract

| Layer | Fixed choice |
| --- | --- |
| Starting point | Every condition begins from the same saved checkpoint `M0`. |
| Candidate pool | Save 64 stochastic draws per source problem. `K=8` is the prefix of `K=64`. |
| Filtering | Compare no update, oracle SFT, exact filtering, and a rate-matched random filter. A useful-noisy or anti-informative filter is earned only after the core comparison works. |
| Update | One rejection-SFT/LoRA update. Prompt, optimizer steps, token budget, and seed remain fixed. |
| Evaluation | Fresh, disjoint held-out manifest; exactly 16 samples per task (`K_eval = 16`). |
| Logging | Parser status, realised filter rates, selected purity, source-problem coverage, duplicates, selected tokens, and per-source repetition. |

The saved pool prevents filter conditions from receiving different raw samples.
Rate matching and token matching prevent a condition from winning only because
it received more training rows or tokens.

## Outcome Labels

All labels use the same held-out manifest, fixed `K_eval = 16`, and a
predeclared paired uncertainty procedure.

- **Operational expansion:** `pass@16` rises and net fixed-budget frontier
  gains (newly solved tasks) exceed losses with positive paired evidence.
- **Sharpening:** `pass@1` rises while `pass@16` has no reliable gain.
- **Collapse:** `pass@16` has a reliable decline.
- **Inconclusive:** uncertainty, churn, or a failed calibration gate prevents a
  label.

"Expansion" is finite-budget evidence only; it never proves that the base
model assigned zero probability to a solution.

## Four Next Steps

1. **Agree the smallest build together.** Settle the task/split, model and
   compute budget, core controls, and success rule before adding variants.
2. **Prepare one shared core.** Keep this README and the frozen manifests
   current. If code helps, add only a generator, exact checker, and the smallest
   directory shape:

   ```text
   src/                 # generator + checker
   scripts/dry_run.py   # one clear entry point
   data/manifests/      # frozen source/test split
   ```

3. **Sketch one dry run.** Its whole path is `generate -> verify -> update ->
   evaluate`; it saves the candidate pool and writes one structured log. No
   framework, future variants, or abstraction layer yet.
4. **Review and freeze together.** Everyone reviews the README and this small
   path. Incorporate only necessary changes, then calibrate and run the core
   comparison. If the oracle gate fails, repair the setup or stop the model grid.

## Repository Contents

```text
README.md                                # Canonical review protocol
presentation/verifier-bottleneck-kickoff.pdf
presentation/verifier-bottleneck-kickoff.pptx
```

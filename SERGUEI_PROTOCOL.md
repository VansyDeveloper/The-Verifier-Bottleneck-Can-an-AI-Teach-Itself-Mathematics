# Serguei Barannikov protocol: checker, exploration, and composition

This file is the scientific contract for the next experiments. It combines the
mentor definition from the screenshot, the 2026 SMILES proposal, the agreed H2
slide deck, and a full audit of the nine local items under `offtop/`.

## 1. The quantity called exploration

Let `pi` be the fixed baseline policy, `rho` an intervention policy, and
`phi_x(y)=u` a predeclared map from a raw completion to an outcome class. Define

$$
\pi_\phi(u\mid x)=\Pr_{y\sim\pi(\cdot\mid x)}[\phi_x(y)=u],
\qquad
S_\pi^\eta(x)=\{u:\pi_\phi(u\mid x)\geq\eta\},
\qquad \eta=1/m,
$$

and Serguei's directed novel mass

$$
\varepsilon^\eta(x;\rho\Vert\pi)
=\Pr_{y\sim\rho(\cdot\mid x)}[\phi_x(y)\notin S_\pi^\eta(x)]
=\sum_{u:\pi_\phi(u\mid x)<\eta}\rho_\phi(u\mid x).
$$

Temperature, mixture weight, search policy, and KL strength are interventions.
They are not epsilon. Epsilon is their measured response relative to a fixed
baseline support. Because `epsilon(pi || pi)` is generally nonzero, the main
contrast is paired:

$$
\Delta\varepsilon^\eta
=\varepsilon^\eta(\rho\Vert\pi)-\varepsilon^\eta(\pi\Vert\pi).
$$

Keep these symbols separate:

- `K`: candidate budget used to measure discovery;
- `m`: support resolution, with `eta=1/m`;
- `M`: independent baseline samples used to estimate support;
- `G`: GRPO group size.

For `K` iid candidates and a fixed support,

$$
\Pr(\text{at least one novel candidate})=1-(1-\varepsilon^\eta)^K.
$$

Thus changing `K` changes discovery probability, not the per-draw epsilon.

## 2. What the current estimator measures

In the affine sandbox,

$$
\phi_x(y)=
\begin{cases}
(A\bmod p,B\bmod p),&\text{if the answer parses},\\
\bot,&\text{otherwise}.
\end{cases}
$$

`eval/eval_exploration.py` uses independent reference and intervention batches
and reports:

- `epsilon_raw`: all mass outside estimated support, including bottom;
- `epsilon_parseable`: novel parsed answer mass;
- `epsilon_correct`: novel exact-gold mass;
- paired deltas against an independent baseline-policy candidate batch;
- parse rate, pass@1, pass@K, support size, and pooled novel precision.

The support plug-in is

$$
\widehat S_\pi^{1/m}(x)=\{u:N_u\geq\lceil M/m\rceil\},
\qquad N_u\sim\operatorname{Binomial}(M,\pi_u).
$$

It estimates mass outside a random support, not the population hard threshold.
Its expectation and bias are

$$
\mathbb E[\widehat\varepsilon]
=\sum_u\rho_u\Pr(N_u<\lceil M/m\rceil),
$$

$$
\operatorname{Bias}
=\sum_{\pi_u\geq 1/m}\rho_u\Pr(N_u<c)
-\sum_{\pi_u<1/m}\rho_u\Pr(N_u\geq c),
\qquad c=\lceil M/m\rceil.
$$

At the default `M=4K`, `m=K`, a class exactly at the threshold is falsely
excluded roughly 43% of the time for `K=32`. Therefore every claim must repeat
model/sampling seeds and show sensitivity across `m`; the CLI accepts several
`--resolution-m` values while reusing the same generated samples. Its standard
errors are across problems conditional on one sampled support, not full model-
seed or support-estimation uncertainty.

Answer-level `epsilon_correct` supports a claim about finite-budget answer
reachability. It does not identify a new reasoning strategy.

## 3. H2 checker grid

For pre-selection correctness `a`, the accepted-pool accuracy is

$$
a_{\rm accepted}=\frac{\alpha a}{\alpha a+\beta(1-a)},
$$

and

$$
a_{\rm accepted}-a
=\frac{a(1-a)(\alpha-\beta)}{\alpha a+\beta(1-a)}.
$$

The signed checker alignment is `J=alpha-beta`:

- `J>0`: accepted samples are enriched for correctness;
- `J=0`: the idealized selection update is neutral;
- `J<0`: the reward is anti-aligned.

Mutual information is not a signed substitute for `J`. An adversarial checker
with `alpha<beta` can have positive mutual information while pointing learning
in the wrong direction. This corrects an imprecise sentence in the proposal.

The implemented H2 protocol is exactly 25 checker cells

```text
alpha in {1.0, 0.8, 0.6, 0.4, 0.2}
beta  in {0.0, 0.2, 0.4, 0.6, 0.8}
seeds in {0, 1, 2}
```

for 75 runs, with model, task distribution, temperature, `G`, token budget, and
optimization held fixed. The primary noise path uses a deterministic uniform
draw keyed by a unique rollout event id assigned independently of answer text.
The answer-keyed quenched checker is a named legacy ablation only.

Each run records configured and realized TPR/FPR, signed alignment, acceptance,
mutual information, zero-variance groups, held-out pass@1, and an independently
selected base-solved subset. The collector computes 90% and 95% hitting times
per seed. It also exposes a preregisterable plateau diagnostic; the default is
an absolute endpoint change no larger than 0.02 over the last five evaluations.

The 600-step value is a minimum horizon, not evidence of convergence. Final H2
claims require `--require-plateau` or a declared longer horizon. Same-`J`
FN-heavy and FP-heavy cells test finite-time error asymmetry.

The RTX 5070 wrapper is deliberately a different estimand: it covers all 25
cells once, adapts a short horizon to a five-hour planning budget, and then
measures epsilon. It is for pipeline validation and effect-size scouting only;
it does not replace the three-seed, 600-step protocol above.

The M1 Pro wrapper first times one complete Transformers/MPS cell. It runs all
25 cells only when that measured cost fits the guarded budget; otherwise it
runs the three declared sentinels $(1,0)$, $(0.6,0.6)$, and $(0.2,0.8)$ for
$J>0$, $J=0$, and $J<0$. This fallback is a phase-boundary diagnostic, not a
completed checker grid. Both modes keep $G$, temperature, task distribution,
and candidate budget fixed and reserve time for the epsilon sweep. The budget is
used only for pre-run sizing: no active training or evaluation process is killed
when the target time is crossed.

The `J=0` phase boundary is already derived and tested in RLV-epsilon-R
(arXiv:2601.04411). Consequently, a plain alpha-beta phase diagram is a
validation/replication. The useful additions here are full error-factorization,
finite-time speed, plateau/ceiling behavior, base-solved change, and later the
interaction with measured exploration.

## 4. Exploration interventions at fixed K

The first two interventions are:

1. temperature sweep;
2. iid policy mixture

$$
\rho_w=(1-w)\pi_{T_0}+w\pi_{T_{\rm high}}.
$$

For fixed baseline support, epsilon is linear in the mixture weight. This is a
useful implementation check. Mixture is initially evaluation-only: naively
mixing an external proposer into GRPO would no longer be ordinary on-policy
sampling and would require likelihood/importance-weight treatment.

The sequence of experiments is:

1. full H2 grid at one fixed sampler;
2. epsilon sweep with exact checker and fixed `K`;
3. only then a sparse cross of selected `J` values with low/high measured
   epsilon.

Do not launch a full 25-by-temperature training product before these controls.

The primary reporting surface is deliberately only three figures:

1. a full $(\alpha,\beta)$ heatmap of held-out final-minus-initial gain, or an
   explicitly cell-counted partial diagnostic, with the $\alpha=\beta$ boundary
   shown explicitly;
2. gain against realized signed alignment (falling back to configured $J$), so
   finite-sample checker behavior is not silently replaced by its target value;
3. parseable $\Delta\widehat\varepsilon^{1/m}$ against pass@1 at fixed $K$,
   separated by temperature and policy-mixture interventions and repeated over
   all requested resolutions.

Raw epsilon is excluded from the main figure because it counts $\bot$. A
one-seed five-hour pilot is labeled as having no seed-level uncertainty, and
the epsilon bars are problem-level standard errors conditional on the sampled
support.

## 5. Composition claim boundary

The current affine-fold sandbox does not test H3. It has one requested operation
and one correct answer class. A rise in `epsilon_correct` is new answer-level
reachability, not proof that learned atoms were composed.

A minimum H3 experiment must:

- train on two atomic operations 50/50;
- never train or reward either ordered composition;
- evaluate atoms and both held-out orders at fixed `K` before and after RL;
- use exact verification, equal rollout/token budgets, and at least three seeds;
- include control, atom-A-only, atom-B-only, and joint A+B checkpoints for a
  causal interaction claim;
- audit traces as primitive, valid sequential/macro, valid parallel, spurious,
  invalid-final-only, or unparseable;
- test whether a valid discovered strategy is reused on new instances.

Without trace validity and reuse, the permitted wording is “functional
generalization to held-out compositions,” not “a new compositional skill.” The
polynomial operators on presentation slide 10 are the agreed full H3 target.
The local `plan_for_dataset.md` affine A/B/AB design is a cheaper precursor, but
it is not the same benchmark and its duplicate-`FINAL` parser rule must first be
made internally consistent.

## 6. Complete local-corpus audit

| Local item | Status | What it contributes | What it does not establish |
|---|---|---|---|
| `2504.13837v5.pdf`, Yue et al., NeurIPS 2025 oral | Peer-reviewed | Short/current RLVR often raises pass@1 while large-K coverage shrinks; motivates sharpening vs reach | No universal impossibility theorem, checker grid, or controlled composition |
| `2505.24864v1.pdf`, ProRL, NeurIPS 2025 | Local v1 preprint; later peer-reviewed | Long, multi-component RL sometimes expands finite-K boundaries, especially from weak nonzero competence | Does not isolate exploration; recipe adds data and costs about 16k H100-hours |
| `2601.04411v1.pdf`, RLV-epsilon-R | Preprint | Derives the `J=TPR-FPR` boundary, support invariance, and mode concentration | Mean-field assumptions; alpha-beta boundary alone is not our novelty |
| `2607.07646v1.pdf`, Abdulsalam et al. | ICML 2026 workshop poster | Auditable primitive/macro/parallel/spurious taxonomy; valid strategy reuse; selectivity over raw novelty | Small synthetic grammar; no noisy checker or scalar-epsilon causality |
| `s41586-025-09833-y.pdf`, AlphaProof | Nature 2026 | Exact Lean checking plus tree search, progressive budgets, and large-scale RL | Search/RL/data effects are entangled and compute is incomparable to one 5070 |
| DeepMind IMO blog PDF | Corporate primary announcement | Reports 28/42 on IMO 2024 and system overview | Not independent evidence; superseded methodologically by the Nature paper |
| `SMILES2026Barannikov_...pdf` | Two-page project proposal | The two-dial H1/H2/H3 program and accepted-pool formula | Does not contain the screenshot epsilon estimator or completed evidence |
| `verifier_bottleneck_guide_ru (1).pdf` | Internal AI-generated synthesis | Useful separation of availability, selection, update, and iid event noise | Not an independent scientific source; cites papers absent from the corpus |
| `plan_for_dataset.md` | Internal implementation spec | Deterministic A/B/AB generation, exact oracle, OOD and adversarial splits | No training/results/epsilon estimator; trace checker is only a stub |

The corpus supports this project statement:

> An exploration intervention can move probability mass into solution or
> strategy classes below a baseline-policy reachability threshold; a positively
> aligned verifier can then retain some of that mass and improve held-out
> composition.

It does not support “RL creates mathematically new nonzero support,” “larger raw
epsilon causes composition,” or “consumer-GPU experiments already generalize to
GSM8K/olympiad mathematics.”

## 7. Public records for the local literature

- Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning
  Capacity in LLMs Beyond the Base Model?*, NeurIPS 2025:
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html>
- ProRL, NeurIPS 2025:
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/1a22b912945fb7c0bdd079e792b31b6f-Abstract-Conference.html>
- *Rate or Fate? RLV-epsilon-R*: <https://arxiv.org/abs/2601.04411>
- Abdulsalam et al., compositional-strategy study:
  <https://openreview.net/forum?id=axxZ7UfwZo>
- AlphaProof, *Nature* 2026:
  <https://www.nature.com/articles/s41586-025-09833-y>

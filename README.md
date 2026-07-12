# The Verifier Bottleneck

> [!WARNING]
> This repository contains exploratory research artifacts. Some experiments and
> figures may be obsolete or based on inconsistent pilot settings. Before using
> the results for serious follow-up work, rerun the key experiments with one
> clean protocol. The sandbox is small, so reproducing them should not take long.

Research code for studying when verifier-guided self-improvement sharpens,
expands, or degrades mathematical ability.

The controlled task is composition of affine functions over finite fields:

$$
g_i(x) = a_i x + b_i \pmod p,
\qquad
(g_n \circ \cdots \circ g_1)(x) = Ax + B \pmod p.
$$

The verifier accepts a correct answer with probability $\alpha$ and an incorrect
answer with probability $\beta$. This makes verifier quality directly
controllable while correctness remains exactly measurable.

## Current evidence

- With an exact verifier, offline SFT on a frozen candidate pool mainly improves
  pass@1, while on-policy GRPO improves both pass@1 and pass@64 on fresh tasks.
- In the initial H2 pilot, $\alpha > \beta$ improved held-out accuracy,
  $\alpha = \beta$ stalled, and $\alpha < \beta$ degraded it. This pilot used
  Qwen3-1.7B-Base, depth 2, one seed, and $\alpha + \beta = 1$.
- Most later sandbox experiments use **Qwen/Qwen3-0.6B-Base**, depths 2-5,
  train primes `{5, 7, 13, 19, 23, 29}`, and held-out primes `{11, 17}`.

The current figures are in [`results/figures`](results/figures), compact data in
[`results/data`](results/data), and model traces in
[`results/traces`](results/traces). Raw server dumps are local and excluded from
Git via `results_remote/`.

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[analysis]'
```

For GPU training and vLLM evaluation:

```bash
pip install -e '.[train]'
```

PyTorch and vLLM compatibility depends on the CUDA driver. The experiments were
run with 8 GPUs; adjust `NPROC` and `CUDA_VISIBLE_DEVICES` for your machine.

## Run

Generate a deterministic sample task:

```bash
python -c "from modcomp.gen import make_dataset; print(make_dataset(1, seed=0)[0])"
```

Train with an exact or noisy verifier:

```bash
NPROC=8 bash scripts/run_train.sh 1.0 0.0 --model Qwen/Qwen3-0.6B-Base
NPROC=8 bash scripts/run_train.sh 0.6 0.4 --model Qwen/Qwen3-0.6B-Base
```

Run the checker sweep or curriculum:

```bash
bash scripts/sweep.sh
bash scripts/curriculum.sh
```

Evaluate pass@k:

```bash
python eval/eval_passk.py \
  --model runs/<run-name>/final \
  --k 64 --n-problems 200 --split eval \
  --prime-max 29 --k-min 2 --k-max 5 --style compact
```

Rebuild the main plots from local run logs:

```bash
python analysis/h2_checker_noise/collect_results.py
python analysis/h2_checker_noise/make_plots.py
```

## Repository layout

- `modcomp/` - task generator and noisy verifier
- `training/` - GRPO, offline-pool generation, and SFT
- `eval/` - pass@k evaluation and shard merging
- `analysis/` - analysis grouped by experiment series
- `scripts/` - experiment launchers
- `results/` - curated public figures, data, and traces
- `presentation/` - agreed research plan

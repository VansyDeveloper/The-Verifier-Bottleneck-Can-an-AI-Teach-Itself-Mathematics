# The Verifier Bottleneck

> [!WARNING]
> This repository contains exploratory research artifacts. Some experiments and
> figures may be obsolete or based on inconsistent pilot settings. Before using
> the results for serious follow-up work, rerun the key experiments with one
> clean protocol. Individual sandbox cells are small, but the full 75-run H2
> grid is still a substantial compute job.

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

Python 3.10--3.13 is supported; Python 3.12 is the recommended reproducible
choice on both Apple Silicon and Linux.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone --branch artem_branch --single-branch \
  https://github.com/VansyDeveloper/The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics.git
cd The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --locked --extra analysis
```

For training and local model evaluation:

```bash
uv sync --locked --extra analysis --extra train
```

The lock is platform-aware: macOS gets PyTorch/Transformers with MPS and no
vLLM; Linux gets the pinned CUDA 12.9 PyTorch and vLLM stack.

### macOS Sequoia 15.6.1 + Apple M1 Pro 32 GB (configured local target)

No VM, Docker, CUDA, MLX, or source build of vLLM is required. First request
Apple's Command Line Tools and wait for their installer to finish:

```bash
xcode-select --install
```

Then install `uv` and clone under your normal home directory:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone --branch artem_branch --single-branch \
  https://github.com/VansyDeveloper/The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics.git
cd The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --locked --extra analysis --extra train
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available()); assert torch.backends.mps.is_available()"
python -m unittest discover -s tests -v
DRY_RUN=1 bash scripts/smoke_m1.sh
```

Qwen3-0.6B is the configured local model. Training uses TRL's ordinary
`model.generate()` path with LoRA and MPS fp16; epsilon evaluation uses the same
Transformers/MPS backend. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the launchers
for unsupported Metal operations. CPU fallback can slow a run but preserves the
pipeline. Dependency resolution and command dry-runs are verified here; actual
MPS timing is measured and cached on the first launch on the Mac.

### Windows 10 + desktop RTX 5070 12 GB (optional CUDA path)

The supported path is WSL2, not native Windows. `wsl --install` requires Windows
10 version 2004 / build 19041 or newer. In an Administrator PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

After restarting, enter Ubuntu. Keep the repository under the Linux home
directory (not `/mnt/c`) and set it up with Python 3.12:

```bash
sudo apt update
sudo apt install -y git python3.12-venv
cd ~
git clone --branch artem_branch --single-branch \
  https://github.com/VansyDeveloper/The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics.git
cd The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv sync --locked --extra analysis --extra train
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
python -m unittest discover -s tests -v
DRY_RUN=1 bash scripts/sweep.sh | grep -c '^START h2_'
```

Install the latest production NVIDIA driver on Windows before these commands;
do not install a Linux display driver inside WSL2. The locked training stack is
`torch==2.10.0+cu129`, `trl==1.5.1`, and `vllm==0.18.0`. A separate CUDA Toolkit
is not needed for the prebuilt wheels. The device check should end in `(12, 0)`,
and the final command must print `75` before the full H2 sweep is used.

## Run

### Start here on the M1 Pro

The intended first real commands are the small smoke test and then the
calibrated pilot; do not start the paper-scale commands below first:

```bash
bash scripts/smoke_m1.sh
bash scripts/run_5h_m1.sh
```

Generate a deterministic sample task:

```bash
python -c "from modcomp.gen import make_dataset; print(make_dataset(1, seed=0)[0])"
```

Manual one-step training examples with an exact or noisy verifier:

```bash
NPROC=1 bash scripts/run_train.sh 1.0 0.0 --model Qwen/Qwen3-0.6B-Base \
  --lora --num-generations 2 --per-device-batch 2 --max-completion-length 64 \
  --max-steps 1 --train-size 512 --eval-size 8 --eval-steps 999999 --no-save
NPROC=1 bash scripts/run_train.sh 0.6 0.4 --model Qwen/Qwen3-0.6B-Base \
  --lora --num-generations 2 --per-device-batch 2 --max-completion-length 64 \
  --max-steps 1 --train-size 512 --eval-size 8 --eval-steps 999999 --no-save
```

On macOS the launcher uses Transformers/MPS; on CUDA Linux it uses colocated
vLLM unless `--generation-backend transformers` is requested.

The default `iid` checker uses a unique answer-independent rollout event id, so
the Bernoulli draw is reproducible but repeated identical completions do not get
a permanently fixed verdict. `--noise-mode hashed` is retained only as a legacy
quenched-noise ablation.

Inspect the paper-scale checker sweep without starting it:

```bash
DRY_RUN=1 bash scripts/sweep.sh
```

The real `bash scripts/sweep.sh` and `bash scripts/curriculum.sh` commands are
long research runs. They are not the first-run path for the M1 Pro.

The H2 sweep implements the presentation protocol: the full $5\times5$
$(\alpha,\beta)$ grid, three seeds, and at least 600 steps with temperature and
the number of generations held fixed. Each run uses one half of a step-0 sample
to select base-solved held-out problems, then measures pass@1 change on independent
baseline/final samples. This avoids calling finite-$K$ nonrediscovery alone
"forgetting." It also writes the realized TPR, FPR, signed alignment, mutual
information, acceptance rate, and zero-variance-group rate. Raise `STEPS` if
curves have not plateaued; a restart skips
cells with the same configuration/code fingerprint and a `.done` marker.
Use environment variables to run a pilot cell or split the grid across machines:

```bash
ALPHAS="1.0" BETAS="0.0" SEEDS="0" STEPS=10 DRY_RUN=1 bash scripts/sweep.sh
```

For the M1 Pro, first run the native smoke test:

```bash
bash scripts/smoke_m1.sh
```

It checks MPS, runs all unit tests, downloads Qwen3-0.6B, times one complete
one-step LoRA cell, runs a tiny fixed-$K$ epsilon sweep, and writes
`results/figures/smoke_m1/epsilon_frontier.{pdf,png}`. The first run depends on
download speed; later runs reuse the model and timing. It has no artificial
timeout.

Then start the five-hour-target M1 pilot:

```bash
bash scripts/run_5h_m1.sh
```

The wrapper uses the measured complete-cell time with a 1.4 safety factor. If
all 25 one-seed cells fit, it runs the full $5\times5$ H2 grid; otherwise it runs
the declared diagnostic cells `(1.0,0.0)`, `(0.6,0.6)`, and `(0.2,0.8)` for
$J>0$, $J=0$, and $J<0$. It then spends the reserved hour on Serguei's fixed-$K$
epsilon sweep and the three core figures. Five hours is the scheduling target,
not a kill boundary: the target chooses the mode and step count before training,
but every started training or evaluation process is allowed to finish naturally,
however long it takes. There is no watchdog or artificial timeout.
On macOS the launchers automatically run under the built-in `caffeinate -i`, so
idle sleep does not pause an overnight job. Keep the Mac connected to power and
leave the lid open; closing the lid can still suspend it.
Use `FORCE_CALIBRATION=1` after changing macOS, PyTorch, the model, or precision.
H2 data and figures are under `results/{data,figures}/pilot5h_m1_<session>/`,
with a matching `results/data/exploration_5h_m1_<session>.json`. The script
prints `mode=full-grid` or `mode=three-cell-diagnostic` and never presents the
latter as the full H2 grid. Either mode is effect scouting, not the 75-run,
600-step paper protocol.

To validate command construction without loading a model:

```bash
DRY_RUN=1 bash scripts/run_5h_m1.sh
```

This dry-run prints the 25-cell command surface; it does not predict whether
runtime calibration will select the full grid or the three-cell diagnostic.
Every cell streams output to the terminal and `logs/<run>.log`; TensorBoard
events remain under `runs/<run>/` while the overnight job continues.

The following RTX/WSL launchers remain available as an optional faster path.

First run a single-GPU 12 GB smoke test under WSL2:

```bash
bash scripts/smoke_5070.sh
```

It runs the unit tests, checks that the paper grid has 75 runs, caches the model,
trains one warm one-step LoRA cell, records its wall time, and runs a tiny
epsilon measurement. It also writes the smoke-test `epsilon_frontier` under
`results/figures/smoke_5070/` and rejects an 8 GB laptop variant instead of
silently using the documented 12 GB desktop profile.
The smoke script records timing but does not impose a runtime cap.
To inspect commands without loading a model, use
`DRY_RUN=1 bash scripts/smoke_5070.sh`.

For the five-hour-target pass, run:

```bash
bash scripts/run_5h_5070.sh
```

This wrapper adds a five-step calibration, estimates model-startup and marginal
step time separately, and chooses at most 25 steps. It covers all 25
`(alpha,beta)` cells once and uses a planned 55-minute epsilon reserve. The
five-hour value sizes the run; it never terminates an active process.
`DRY_RUN=1 bash scripts/run_5h_5070.sh` must print 25 `START` lines.
It is a one-seed, short-horizon pilot and cannot support the paper-level phase
diagram by itself. No honest configuration can fit the preregistered 75 runs x
600 steps into five hours on one RTX 5070; those remain a separate long run.
Run the smoke test first so model weights are cached and the calibration belongs
to this driver/software profile. Use `FORCE_CALIBRATION=1` after changing the
driver, lockfile, model, or memory mode. Disable Windows sleep for the run.
Cells are attempted independently so a natural failure such as OOM does not
cancel later cells; incomplete run directories are moved under
`runs/incomplete/`, the session returns nonzero, and they are never mixed with
a later analysis. Successful pilot summaries are written under
`results/{data,figures}/pilot5h_<session>/`; epsilon is written to a matching
session-tagged JSON file. Plotting is CPU-only and runs automatically at the end.
It produces exactly three core figures, each as vector PDF and 300-DPI PNG:
`h2_gain_heatmap`, `h2_gain_vs_alignment`, and `epsilon_frontier`.

If the colocated smoke run exhausts 12 GB during training, retry with
`SLEEP_MODE=1 bash scripts/smoke_5070.sh`. Sleep mode trades speed and host-RAM
traffic for a lower training-phase VRAM peak; it cannot fix an OOM while vLLM
is first constructing its engine. Leave it off when the ordinary smoke test
already fits.

Then benchmark one 25-step cell:

```bash
NPROC=1 ALPHAS="1.0" BETAS="0.0" SEEDS="0" STEPS=25 GENERATIONS=4 \
  bash scripts/sweep.sh --lora --per-device-batch 4 --grad-accum 8 \
  --max-completion-length 256 --vllm-gpu-memory-utilization 0.20
```

If that succeeds, removing `ALPHAS`, `BETAS`, and `SEEDS` schedules all 75 runs:

```bash
NPROC=1 STEPS=600 GENERATIONS=4 bash scripts/sweep.sh \
  --lora --per-device-batch 4 --grad-accum 8 \
  --max-completion-length 256 --vllm-gpu-memory-utilization 0.20
```

The full grid is approximately `75 * 600 / 25 = 1800` times the wall time of
the 25-step benchmark, before overhead. Do not start it blindly on one desktop.
Qwen3-0.6B-Base is the recommended local model; the earlier 1.7B pilot is not
the default 12 GB target.

Paper-scale/long-run Barannikov effective-support measurement at fixed $K$
(do not use this as the first M1 run):

```bash
python eval/eval_exploration.py \
  --model Qwen/Qwen3-0.6B-Base --k 32 --reference-samples 128 \
  --resolution-m 16 32 64 \
  --out results/data/exploration_sweep.json
```

Here temperature and mixture weight are interventions; $\varepsilon$ is the
measured response. `K` is the fixed candidate budget, `m` defines
$\eta=1/m$, `M=reference-samples` estimates the baseline support, and GRPO's
group size `G` is a separate training parameter. The output reports a
finite-sample plug-in $\widehat\varepsilon^{1/m}_{\rho\mid\pi}$ at all requested
resolutions using the same sampled candidates. Raw novelty
includes the unparseable $\bot$ class, so use parseable and correct novel mass
as the primary outcomes. The table includes paired deltas and problem-level
standard errors. The plug-in support is random and threshold-biased near
$\pi(u\mid x)=1/m$; repeat seeds and report resolution sensitivity. In this
affine task it is an answer-reachability metric, not evidence by itself of a new
strategy.

Small inference-only epsilon smoke test for M1/MPS:

```bash
python eval/eval_exploration.py --model Qwen/Qwen3-0.6B-Base \
  --backend transformers --device mps --generation-batch-size 4 \
  --n-problems 4 --k 4 --reference-samples 16 --resolution-m 4 8 \
  --max-tokens 96 \
  --out results/data/exploration_smoke.json
```

Render its exploration figure without rerunning the model:

```bash
python analysis/plot_core_results.py \
  --exploration results/data/exploration_smoke.json \
  --out results/figures/exploration_smoke
```

To check the plotting stack in seconds, without a GPU or experiment data:

```bash
python analysis/plot_core_results.py --demo --out results/figures/demo
```

The command prints `DEMO ONLY`; those synthetic values test layout only and must
not be reported as results.

Optional bulk pass@k evaluation currently uses the CUDA/vLLM path:

```bash
python eval/eval_passk.py \
  --model runs/<run-name>/final \
  --k 64 --n-problems 200 --split eval \
  --prime-max 29 --k-min 2 --k-max 5 --style compact
```

Collect H2 data and rebuild the main figures from local run logs:

```bash
python analysis/h2_checker_noise/collect_results.py
python analysis/plot_core_results.py \
  --h2-summary results/data/checker_grid_summary.json \
  --exploration results/data/exploration_sweep.json \
  --out results/figures/core
```

If more than one configuration fingerprint exists, the collector refuses to
mix them and prints the exact `--run-prefix h2_c<id>_` choices. It also requires
all 25 cells, three seeds per cell, step 0, at least 600 steps, and base-solved
change plus checker-diagnostic metrics. Hitting times are computed per seed, not
from the averaged curve. For a final convergence claim, use
`--require-plateau`; its default diagnostic is an endpoint change of at most
0.02 over the last five evaluations. For a short pilot, pass both
`--allow-partial-grid` and `--allow-short-horizon`.

The heatmap reports final-minus-initial held-out gain on the canonical $5\times5$
checker axes. Missing pilot cells stay grey, and a diagnostic fallback is titled
with its explicit cell count rather than presented as a phase map. The alignment
plot uses realized TPR−FPR when diagnostics exist and falls back to configured
$J=\alpha-\beta$; a one-seed pilot is explicitly labeled as having no uncertainty
estimate. The epsilon frontier plots parseable novel mass against pass@1
separately for temperature and policy mixture at fixed $K$ and all requested
resolutions. It deliberately excludes raw novelty, which counts the unparseable
$\bot$ outcome. Answer-level novelty is reachability evidence, not by itself
evidence of a new compositional reasoning strategy.

The exact mathematical contract and the local-literature implications are in
[`SERGUEI_PROTOCOL.md`](SERGUEI_PROTOCOL.md).

## Repository layout

- `modcomp/` - task generator and noisy verifier
- `training/` - GRPO, offline-pool generation, and SFT
- `eval/` - pass@k evaluation and shard merging
- `analysis/` - analysis grouped by experiment series
- `scripts/` - experiment launchers
- `results/` - curated public figures, data, and traces
- `presentation/` - agreed research plan

# Model reasoning traces

Rollouts logged during GRPO (`log_completions=True`). The prompt shows two solved few-shot examples then the problem to solve; only the final problem block is reproduced here. Reward is the noisy checker's verdict; `true_correct` is the noiseless ground truth.

## Perfect checker (α=1, β=0) — reasoning tightens over training

### Early correct solve

*step 1 · checker_reward=1 · true_correct=1*

```
Problem:
p = 7
g_1(x) = 5*x + 6
g_2(x) = 4*x + 0
Solution:
g_2(g_1(x)) = 4*(5*x + 6) = 20*x + 24 = 6*x + 3 (mod 7).
Answer: A=6, B=3

p = 11
g_1(x) = 9*x + 3
g_2(x) = 2*x + 9
g_3(x) = 8*x + 4

Problem:
```

### Late correct solve #1

*step 240 · checker_reward=1 · true_correct=1*

```
Problem:
p = 19
g_1(x) = 9*x + 18
g_2(x) = 4*x + 1
Solution:
g_2(g_1(x)) = 4*(9*x + 18) + 1 = 36*x + 72 + 1 = 36*x + 73.
Answer: A=36, B=73
```


### Late correct solve #2

*step 240 · checker_reward=1 · true_correct=1*

```
Problem:
p = 19
g_1(x) = 9*x + 18
g_2(x) = 4*x + 1
Solution:
g_2(g_1(x)) = 4*(9*x + 18) + 1 = 36*x + 72 + 1 = 36*x + 73.
36 mod 19 = 36 - 19*1 = 36 - 19 = 17.
Answer: A=17, B=73
```


### Late incorrect (residual failure mode)

*step 241 · checker_reward=0 · true_correct=0*

```
Problem:
p = 19
g_1(x) = 15*x + 12
g_2(x) = 16*x + 9
Solution:
g_2(g_1(x)) = 16*(15*x + 12) + 9 = 240*x + 192 + 9 = 240*x + 101.
240 mod 19 = 240 - 19*12 = 240 - 228 = 12.
Answer: A=12, B=101
```


## Break-even checker (α=β=0.5) — reward decoupled from truth

### WRONG answer the checker still rewarded

*step 125 · checker_reward=1 · true_correct=0*

```
Problem:
p = 11
g_1(x) = 5*x + 1
g_2(x) = 5*x + 5
Solution:
g_2(g_1(x)) = 5*(5*x + 1) + 5 = 25*x + 10 = 1*x + 1 = x + 1 (mod 11).
Answer: A=1, B=1

Problem:
```


### CORRECT answer the checker rejected

*step 124 · checker_reward=0 · true_correct=1*

```
Problem:
p = 7
g_1(x) = 1*x + 6
g_2(x) = 3*x + 4
Solution:
g_2(g_1(x)) = 3*(x + 6) + 4 = 3*x + 18 = 3*x + 1 (mod 7).
Answer: A=3, B=1
```


## Curriculum top rung (p≤97, perfect judge) — hardest modulus solved

### Solve #1

*step 140 · checker_reward=1 · true_correct=1*

```
Problem:
p = 89
g_1(x) = 75*x + 43
g_2(x) = 19*x + 75
Solution:
g_2(g_1(x)) = 19*(75*x + 43) + 75 = 1425*x + 817 + 75 = 1425*x + 892.
Answer: A=1425, B=892

Problem:
```


### Solve #2

*step 140 · checker_reward=1 · true_correct=1*

```
Problem:
p = 89
g_1(x) = 75*x + 43
g_2(x) = 19*x + 75
Solution:
g_2(g_1(x)) = 19*(75*x + 43) + 75 = 1425*x + 817 + 75 = 1425*x + 892.
Answer: A=1425, B=892

Problem:
```

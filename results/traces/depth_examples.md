# Deep-composition reasoning traces (compact format, up to k=10)

Rollouts from the compact-format run (`--style compact`, k∼U(2,10), p≤23, perfect
judge, 300 steps). The model carries a running accumulator `(A, B)` and folds in one
function per line, reducing mod p at each step. Both examples are correct
(`true_correct=1`) and show clean depth-9 / depth-10 compositions.

### Depth k=9, correct (step 255)

```
Problem:
p = 7
g_1(x) = 6*x + 6
g_2(x) = 2*x + 4
g_3(x) = 4*x + 3
g_4(x) = 3*x + 6
g_5(x) = 2*x + 6
g_6(x) = 5*x + 5
g_7(x) = 4*x + 3
g_8(x) = 1*x + 2
g_9(x) = 3*x + 2
Solution:
Start: A=1, B=0
g_1: A=6*1=6, B=6*0+6=6 -> A=6, B=6
g_2: A=2*6=12=5, B=2*6+4=16=2 -> A=5, B=2
g_3: A=4*5=20=6, B=4*2+3=11=4 -> A=6, B=4
g_4: A=3*6=18=4, B=3*4+6=18=4 -> A=4, B=4
g_5: A=2*4=8=1, B=2*4+6=14=0 -> A=1, B=0
g_6: A=5*1=5, B=5*0+5=5 -> A=5, B=5
g_7: A=4*5=20=6, B=4*5+3=23=2 -> A=6, B=2
g_8: A=1*6=6, B=1*2+2=4 -> A=6, B=4
g_9: A=3*6=18=4, B=3*4+2=14=0 -> A=4, B=0
Answer: A=4, B=0
```

### Depth k=10, correct (step 257)

```
Problem:
p = 5
g_1(x) = 3*x + 3
g_2(x) = 2*x + 1
g_3(x) = 4*x + 3
g_4(x) = 1*x + 2
g_5(x) = 2*x + 0
g_6(x) = 4*x + 2
g_7(x) = 3*x + 3
g_8(x) = 2*x + 2
g_9(x) = 4*x + 3
g_10(x) = 4*x + 1
Solution:
Start: A=1, B=0
g_1: A=3*1=3, B=3*0+3=3 -> A=3, B=3
g_2: A=2*3=6=1, B=2*3+1=7=2 -> A=1, B=2
g_3: A=4*1=4, B=4*2+3=11=1 -> A=4, B=1
g_4: A=1*4=4, B=1*1+2=3 -> A=4, B=3
g_5: A=2*4=8=3, B=2*3+0=6=1 -> A=3, B=1
g_6: A=4*3=12=2, B=4*1+2=6=1 -> A=2, B=1
g_7: A=3*2=6=1, B=3*1+3=6=1 -> A=1, B=1
g_8: A=2*1=2, B=2*1+2=4 -> A=2, B=4
g_9: A=4*2=8=3, B=4*4+3=19=4 -> A=3, B=4
g_10: A=4*3=12=2, B=4*4+1=17=2 -> A=2, B=2
Answer: A=2, B=2
```

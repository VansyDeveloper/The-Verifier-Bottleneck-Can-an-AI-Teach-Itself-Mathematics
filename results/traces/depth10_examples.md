# Depth-8/10 reasoning traces (compact accumulator, p<=13)

### depth k=8 (solved, held-out instance)

Problem:
```
p = 5
g_1(x) = 1*x + 4
g_2(x) = 2*x + 4
g_3(x) = 2*x + 2
g_4(x) = 4*x + 2
g_5(x) = 1*x + 4
g_6(x) = 1*x + 1
g_7(x) = 4*x + 4
g_8(x) = 2*x + 0
```
Model solution:
```
Start: A=1, B=0
g_1: A=1*1=1, B=1*0+4=4 -> A=1, B=4
g_2: A=2*1=2, B=2*4+4=12=2 -> A=2, B=2
g_3: A=2*2=4, B=2*2+2=6=1 -> A=4, B=1
g_4: A=4*4=16=1, B=4*1+2=6=1 -> A=1, B=1
g_5: A=1*1=1, B=1*1+4=5=0 -> A=1, B=0
g_6: A=1*1=1, B=1*0+1=1 -> A=1, B=1
g_7: A=4*1=4, B=4*1+4=8=3 -> A=4, B=3
g_8: A=2*4=8=3, B=2*3+0=6=1 -> A=3, B=1
Answer: A=3, B=1
```

### depth k=10 (solved, held-out instance)

Problem:
```
p = 5
g_1(x) = 1*x + 1
g_2(x) = 4*x + 4
g_3(x) = 3*x + 1
g_4(x) = 4*x + 0
g_5(x) = 4*x + 2
g_6(x) = 2*x + 0
g_7(x) = 3*x + 2
g_8(x) = 1*x + 0
g_9(x) = 3*x + 4
g_10(x) = 3*x + 1
```
Model solution:
```
Start: A=1, B=0
g_1: A=1*1=1, B=1*0+1=1 -> A=1, B=1
g_2: A=4*1=4, B=4*1+4=8=3 -> A=4, B=3
g_3: A=3*4=12=2, B=3*3+1=10=5 -> A=2, B=5
g_4: A=4*2=8=3, B=4*5+0=20=0 -> A=3, B=0
g_5: A=4*3=12=2, B=4*0+2=2=2 -> A=2, B=2
g_6: A=2*2=4, B=2*2+0=4=4 -> A=4, B=4
g_7: A=3*4=12=2, B=3*4+2=14=4 -> A=2, B=4
g_8: A=1*2=2, B=1*4+0=4=4 -> A=2, B=4
g_9: A=3*2=6=1, B=3*4+4=16=1 -> A=1, B=1
g_10: A=3*1=3, B=3*1+1=4=4 -> A=3, B=4
Answer: A=3, B=4

Extensions:
Express B in terms of A.
For which p is A = B? Which solutions have p > 2?
For which p
```

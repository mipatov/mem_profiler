# mem_profiler

```python
from mem_profiler.profiler import Profiler
prof = Profiler()

prof.gpu_mem()
# (5804.0, 15109.75)

prof.gpu_mem_info() 
# gpu mem : 5840.0/15109.8 mb

report = prof.one_step_report(batch, model,optim,device = DEVICE)
# begin gpu mem : 5804.0/15109.8 mb
# 0.050s forward gpu mem : 13006.0/15109.8 mb
# 1.232s backward gpu mem : 14576.0/15109.8 mb
# 0.025s optimizer_step gpu mem : 14576.0/15109.8 mb

report
```
|index|used\_mem|delta\_mem|delta\_time|
|---|---|---|---|
|begin|5804\.0|0\.0|0\.0|
|forward|13006\.0|7202\.0|0\.04|
|backward|14576\.0|8772\.0|1\.2|
|optim\_step|14576\.0|8772\.0|0\.02|
|end|5840\.0|36\.0|0\.13|
|total|15109\.75|0\.0|1\.396|

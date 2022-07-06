# mem_profiler

```python
from mem_profiler.profiler import Profiler
prof = Profiler()

prof.gpu_mem()
# (5804.0, 15109.75)

prof.gpu_mem_info() 
# gpu mem : 5840.0/15109.8 mb

report = prof.one_step_report(batch, model,optim,device = DEVICE)
# begin gpu mem : 5840.0/15109.8 mb
# forward gpu mem : 13006.0/15109.8 mb
# backward gpu mem : 14576.0/15109.8 mb
# optimizer_step gpu mem : 14576.0/15109.8 mb

report
```
|index|used|delta|
|---|---|---|
|begin|5840\.0|0\.0|
|forward|13006\.0|7166\.0|
|backward|14576\.0|8736\.0|
|optimizer\_step|14576\.0|8736\.0|
|end|5804\.0|-36\.0|

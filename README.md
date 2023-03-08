# metaheuristics-benchmark
Python3 framework for comparative study of recent metaheuristics

# Goals

Our goals are to implement some of the classical as well as the state-of-the-art metaheuristics, create a simple interface that helps researchers
access optimization algorithms as quickly as possible, and share knowledge of the optimization field with everyone without a fee.

# Usage

## Example

```python 
from functions import *
from metaheuristics import *
import numpy as np

dim = 12

cost = Rastrigin(dim)
model = whale_optimization(cost)
state, cost, states, costs = model.optimize()

model.plot()
```

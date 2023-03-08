# metaheuristics-benchmark
Python3 framework for comparative study of recent metaheuristics

* **Free software:** MIT license
* **Python versions:** 3.7.x, 3.8.x, 3.9.x, 3.10.x
* **Dependencies:** numpy, matplotlib

# Goals

Our goals are
- To implement some of the classical as well as the state-of-the-art metaheuristics.
- To implement the IEEE-CEC benchmark functions set.
- Create a simple interface that helps researchers, practitioners and students access optimization algorithms as quickly as possible, evaluate their performances against most common metaheuristics and share knowledge of the optimization field with everyone without a fee.

# Usage

## Example

```python
from functions import *
from metaheuristics import *


cost = Rastrigin(dim=12) 
model = whale_optimization(cost)
best_agent, cost_best_agent, costs = model.optimize()
```

### Get Visualize Figures
```python
model.plot()
```
<p align="center">
  <img alt="Light" src="example1.png" width="45%">
</p>

# Functions

We tried to implement all functions of the IEEE-CEC-2017 benchmark functions, <a src="https://github.com/P-N-Suganthan/CEC2017-BoundContrained">listed here</a>

# Optimization algorithms
For now, available metaheuristics optimization algorithms are
- Simulated Annealing
- Differential Evolution
- Grey Wolf Optimizer
- Whale Optimization
- Equilibrium Optimizer

More are coming soon.

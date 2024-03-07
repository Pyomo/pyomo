# Pyomo MPC

Pyomo MPC is an extension for developing model predictive control simulations
using Pyomo models. Please see the
[documentation](https://pyomo.readthedocs.io/en/stable/contributed_packages/mpc/index.html)
for more detailed information.

Pyomo MPC helps with, among other things, the following use cases:
- Transferring values between different points in time in a dynamic model
(e.g. to initialize a dynamic model to its initial conditions)
- Extracting or loading disturbances and inputs from or to models, and storing
these in model-agnostic, easily JSON-serializable data structures
- Constructing common modeling components, such as weighted-least-squares
tracking objective functions, piecewise-constant input constraints, or
terminal region constraints.

## Citation

If you use Pyomo MPC in your research, please cite the following paper, which
discusses the motivation for the Pyomo MPC data structures and the underlying
Pyomo features that make them possible.
```bibtex
@article{parker2023mpc,
title = {Model predictive control simulations with block-hierarchical differential-algebraic process models},
journal = {Journal of Process Control},
volume = {132},
pages = {103113},
year = {2023},
issn = {0959-1524},
doi = {https://doi.org/10.1016/j.jprocont.2023.103113},
url = {https://www.sciencedirect.com/science/article/pii/S0959152423002007},
author = {Robert B. Parker and Bethany L. Nicholson and John D. Siirola and Lorenz T. Biegler},
}
```

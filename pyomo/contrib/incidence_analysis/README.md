# Incidence Analysis

Tools for constructing and analyzing the incidence graph of variables and
constraints.

These tools can be used to detect whether and (approximately) why the Jacobian
of equality constraints is structurally or numerically singular, which
commonly happens as the result of a modeling error.
See the
[documentation](https://pyomo.readthedocs.io/en/stable/explanation/analysis/incidence/index.html)
for more information and examples.

## Dependencies

Incidence Analysis uses
[NetworkX](https://github.com/networkx/networkx)
to represent incidence graphs. Additionally,
[SciPy](https://github.com/scipy/scipy)
and
[Plotly](https://github.com/plotly/plotly.py)
may be required for some functionality.

## Example

Identifying over-constrained and under-constrained subsystems of a singular
square system:
```python
import pyomo.environ as pyo
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

m = pyo.ConcreteModel()
m.components = pyo.Set(initialize=[1, 2, 3])
m.x = pyo.Var(m.components, initialize=1.0/3.0)
m.flow_comp = pyo.Var(m.components, initialize=10.0)
m.flow = pyo.Var(initialize=30.0)
m.density = pyo.Var(initialize=1.0)

m.sum_eqn = pyo.Constraint(
    expr=sum(m.x[j] for j in m.components) - 1 == 0
)
m.holdup_eqn = pyo.Constraint(m.components, expr={
    j: m.x[j]*m.density - 1 == 0 for j in m.components
})
m.density_eqn = pyo.Constraint(
    expr=1/m.density - sum(1/m.x[j] for j in m.components) == 0
)
m.flow_eqn = pyo.Constraint(m.components, expr={
    j: m.x[j]*m.flow - m.flow_comp[j] == 0 for j in m.components
})

igraph = IncidenceGraphInterface(m, include_inequality=False)
var_dmp, con_dmp = igraph.dulmage_mendelsohn()

uc_var = var_dmp.unmatched + var_dmp.underconstrained
uc_con = con_dmp.underconstrained
oc_var = var_dmp.overconstrained
oc_con = con_dmp.overconstrained + con_dmp.unmatched

print("Overconstrained subsystem")
print("-------------------------")
print("Variables")
for var in oc_var:
    print(f"  {var.name}")
print("Constraints")
for con in oc_con:
    print(f"  {con.name}")
print()

print("Underconstrained subsystem")
print("--------------------------")
print("Variables")
for var in uc_var:
    print(f"  {var.name}")
print("Constraints")
for con in uc_con:
    print(f"  {con.name}")
```
This displays:
```console
Overconstrained subsystem
-------------------------
Variables
  x[1]
  density
  x[2]
  x[3]
Constraints
  sum_eqn
  holdup_eqn[1]
  holdup_eqn[2]
  holdup_eqn[3]
  density_eqn

Underconstrained subsystem
--------------------------
Variables
  flow_comp[1]
  flow
  flow_comp[2]
  flow_comp[3]
Constraints
  flow_eqn[1]
  flow_eqn[2]
  flow_eqn[3]
```

## Citation

If you use Incidence Analysis in your research, we would appreciate you citing
the following paper:
```bibtex
@article{parker2023dulmage,
title = {Applications of the {Dulmage-Mendelsohn} decomposition for debugging nonlinear optimization problems},
journal = {Computers \& Chemical Engineering},
volume = {178},
pages = {108383},
year = {2023},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2023.108383},
url = {https://www.sciencedirect.com/science/article/pii/S0098135423002533},
author = {Robert B. Parker and Bethany L. Nicholson and John D. Siirola and Lorenz T. Biegler},
}
```

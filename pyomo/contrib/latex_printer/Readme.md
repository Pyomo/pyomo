# Pyomo LaTeX Printer

This is a prototype latex printer for Pyomo models.  DISCLAIMER:  The API for the LaTeX printer is not finalized and may have a future breaking change.  Use at your own risk.

## Usage

```python
import pyomo.environ as pyo
from pyomo.contrib.latex_printer import latex_printer

m = pyo.ConcreteModel(name = 'basicFormulation')
m.x = pyo.Var()
m.y = pyo.Var()
m.z = pyo.Var()
m.c = pyo.Param(initialize=1.0, mutable=True)
m.objective    = pyo.Objective( expr = m.x + m.y + m.z )
m.constraint_1 = pyo.Constraint(expr = m.x**2 + m.y**2.0 - m.z**2.0 <= m.c )

pstr = latex_printer(m)
```


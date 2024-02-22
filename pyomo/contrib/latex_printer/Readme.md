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


## Acknowledgement

Pyomo: Python Optimization Modeling Objects  
Copyright (c) 2008-2023  
National Technology and Engineering Solutions of Sandia, LLC  
Under the terms of Contract DE-NA0003525 with National Technology and
Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
rights in this software.

Development of this module was conducted as part of the Institute for
the Design of Advanced Energy Systems (IDAES) with support through the
Simulation-Based Engineering, Crosscutting Research Program within the
U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.

This software is distributed under the 3-clause BSD License.
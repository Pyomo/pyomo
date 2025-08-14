#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

# ---------------------------------------------
# @categories
m = pyo.ConcreteModel()
m.p = pyo.Param(default=10, mutable=False)
m.q = pyo.Param(default=10, mutable=True)
m.x = pyo.Var()
m.y = pyo.Var(initialize=1)
m.y.fixed = True
# @categories
m.pprint()

# ---------------------------------------------
# @named_expression
M = pyo.ConcreteModel()
M.v = pyo.Var()
M.w = pyo.Var()

M.e = pyo.Expression(expr=2 * M.v)
f = M.e + 3  # f == 2*v + 3
M.e += M.w  # f == 2*v + 3 + w
# @named_expression

# ---------------------------------------------
# @cm1
M = pyo.ConcreteModel()
M.x = pyo.Var(range(5))

s = 0
for i in range(5):
    s += M.x[i]

with pyo.linear_expression() as e:
    for i in range(5):
        e += M.x[i]
# @cm1
print(s)
print(e)


# ---------------------------------------------
# @cm2
M = pyo.ConcreteModel()
M.x = pyo.Var(range(5))
M.y = pyo.Var(range(5))

with pyo.linear_expression() as e:
    pyo.quicksum((M.x[i] for i in M.x), start=e)
    pyo.quicksum((M.y[i] for i in M.y), start=e)
# @cm2
print("cm2")
print(e)

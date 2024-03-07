#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *

# ---------------------------------------------
# @categories
m = ConcreteModel()
m.p = Param(default=10, mutable=False)
m.q = Param(default=10, mutable=True)
m.x = Var()
m.y = Var(initialize=1)
m.y.fixed = True
# @categories
m.pprint()

# ---------------------------------------------
# @named_expression
M = ConcreteModel()
M.v = Var()
M.w = Var()

M.e = Expression(expr=2 * M.v)
f = M.e + 3  # f == 2*v + 3
M.e += M.w  # f == 2*v + 3 + w
# @named_expression

# ---------------------------------------------
# @cm1
M = ConcreteModel()
M.x = Var(range(5))

s = 0
for i in range(5):
    s += M.x[i]

with linear_expression() as e:
    for i in range(5):
        e += M.x[i]
# @cm1
print(s)
print(e)


# ---------------------------------------------
# @cm2
M = ConcreteModel()
M.x = Var(range(5))
M.y = Var(range(5))

with linear_expression() as e:
    quicksum((M.x[i] for i in M.x), start=e)
    quicksum((M.y[i] for i in M.y), start=e)
# @cm2
print("cm2")
print(e)

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
# @example1
M = pyo.ConcreteModel()
M.x = pyo.Var(range(100))

# This loop is fast.
e = 0
for i in range(100):
    e = e + M.x[i]

# This loop is slow.
e = 0
for i in range(100):
    e = M.x[i] + e
# @example1
print(e)

# ---------------------------------------------
# @example2
M = pyo.ConcreteModel()
M.p = pyo.Param(initialize=3)
M.q = 1 / M.p
M.x = pyo.Var(range(100))

# The value M.q is cloned every time it is used.
e = 0
for i in range(100):
    e = e + M.x[i] * M.q
# @example2
print(e)

# ---------------------------------------------
# @tree1
M = pyo.ConcreteModel()
M.v = pyo.Var()

e = f = 2 * M.v
# @tree1
print(e)

# ---------------------------------------------
# @tree2
M = pyo.ConcreteModel()
M.v = pyo.Var()

e = 2 * M.v
f = e + 3
# @tree2
print(e)
print(f)

# ---------------------------------------------
# @tree3
M = pyo.ConcreteModel()
M.v = pyo.Var()

e = 2 * M.v
f = e + 3
g = e + 4
# @tree3
print(e)
print(f)
print(g)

# ---------------------------------------------
# @tree4
M = pyo.ConcreteModel()
M.v = pyo.Var()
M.w = pyo.Var()

e = 2 * M.v
f = e + 3

e += M.w
# @tree4
print(e)
print(f)

# ---------------------------------------------
# @tree5
M = pyo.ConcreteModel()
M.v = pyo.Var()
M.w = pyo.Var()

M.e = pyo.Expression(expr=2 * M.v)
f = M.e + 3

M.e += M.w
# @tree5
print(M.e)

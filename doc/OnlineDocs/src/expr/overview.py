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
# @example1
M = ConcreteModel()
M.x = Var(range(100))

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
M = ConcreteModel()
M.p = Param(initialize=3)
M.q = 1 / M.p
M.x = Var(range(100))

# The value M.q is cloned every time it is used.
e = 0
for i in range(100):
    e = e + M.x[i] * M.q
# @example2
print(e)

# ---------------------------------------------
# @tree1
M = ConcreteModel()
M.v = Var()

e = f = 2 * M.v
# @tree1
print(e)

# ---------------------------------------------
# @tree2
M = ConcreteModel()
M.v = Var()

e = 2 * M.v
f = e + 3
# @tree2
print(e)
print(f)

# ---------------------------------------------
# @tree3
M = ConcreteModel()
M.v = Var()

e = 2 * M.v
f = e + 3
g = e + 4
# @tree3
print(e)
print(f)
print(g)

# ---------------------------------------------
# @tree4
M = ConcreteModel()
M.v = Var()
M.w = Var()

e = 2 * M.v
f = e + 3

e += M.w
# @tree4
print(e)
print(f)

# ---------------------------------------------
# @tree5
M = ConcreteModel()
M.v = Var()
M.w = Var()

M.e = Expression(expr=2 * M.v)
f = M.e + 3

M.e += M.w
# @tree5
print(M.e)

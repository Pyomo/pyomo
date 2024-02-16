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

# Sample Problem 1 (Ex 1 from Dynopt Guide)
#
# 	min X2(tf)
# 	s.t.	X1_dot = u			X1(0) = 1
# 		X2_dot = X1^2 + u^2		X2(0) = 0
# 		tf = 1

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()

m.t = ContinuousSet(bounds=(0, 1))

m.x1 = Var(m.t, bounds=(0, 1))
m.x2 = Var(m.t, bounds=(0, 1))
m.u = Var(m.t, initialize=0)

m.x1dot = DerivativeVar(m.x1)
m.x2dot = DerivativeVar(m.x2)

m.obj = Objective(expr=m.x2[1])


def _x1dot(M, i):
    if i == 0:
        return Constraint.Skip
    return M.x1dot[i] == M.u[i]


m.x1dotcon = Constraint(m.t, rule=_x1dot)


def _x2dot(M, i):
    if i == 0:
        return Constraint.Skip
    return M.x2dot[i] == M.x1[i] ** 2 + M.u[i] ** 2


m.x2dotcon = Constraint(m.t, rule=_x2dot)


def _init(M):
    yield M.x1[0] == 1
    yield M.x2[0] == 0
    yield ConstraintList.End


m.init_conditions = ConstraintList(rule=_init)

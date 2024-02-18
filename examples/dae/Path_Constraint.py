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

# Sample Problem 3: Inequality State Path Constraint
# (Ex 4 from Dynopt Guide)
#
#   min x3(tf)
#   s.t.    X1_dot = X2                     X1(0) =  0
#           X2_dot = -X2+u                  X2(0) = -1
#           X3_dot = X1^2+x2^2+0.005*u^2    X3(0) =  0
#           X2-8*(t-0.5)^2+0.5 <= 0
#           tf = 1
#

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()
m.tf = Param(initialize=1)
m.t = ContinuousSet(bounds=(0, m.tf))

m.u = Var(m.t, initialize=0)
m.x1 = Var(m.t)
m.x2 = Var(m.t)
m.x3 = Var(m.t)

m.dx1 = DerivativeVar(m.x1, wrt=m.t)
m.dx2 = DerivativeVar(m.x2, wrt=m.t)
m.dx3 = DerivativeVar(m.x3, wrt=m.t)

m.obj = Objective(expr=m.x3[m.tf])


def _x1dot(m, t):
    if t == 0:
        return Constraint.Skip
    return m.dx1[t] == m.x2[t]


m.x1dotcon = Constraint(m.t, rule=_x1dot)


def _x2dot(m, t):
    if t == 0:
        return Constraint.Skip

    return m.dx2[t] == -m.x2[t] + m.u[t]


m.x2dotcon = Constraint(m.t, rule=_x2dot)


def _x3dot(m, t):
    if t == 0:
        return Constraint.Skip

    return m.dx3[t] == m.x1[t] ** 2 + m.x2[t] ** 2 + 0.005 * m.u[t] ** 2


m.x3dotcon = Constraint(m.t, rule=_x3dot)


def _con(m, t):
    return m.x2[t] - 8 * (t - 0.5) ** 2 + 0.5 <= 0


m.con = Constraint(m.t, rule=_con)


def _init(m):
    yield m.x1[0] == 0
    yield m.x2[0] == -1
    yield m.x3[0] == 0


m.init_conditions = ConstraintList(rule=_init)

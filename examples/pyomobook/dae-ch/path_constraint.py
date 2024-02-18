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
# @import:
import pyomo.environ as pyo
import pyomo.dae as dae

# @:import

m = pyo.ConcreteModel()

# @contset:
m.tf = pyo.Param(initialize=1)
m.t = dae.ContinuousSet(bounds=(0, m.tf))
# @:contset

# @vardecl:
m.u = pyo.Var(m.t, initialize=0)
m.x1 = pyo.Var(m.t)
m.x2 = pyo.Var(m.t)
m.x3 = pyo.Var(m.t)

m.dx1 = dae.DerivativeVar(m.x1, wrt=m.t)
m.dx2 = dae.DerivativeVar(m.x2, wrt=m.t)
m.dx3 = dae.DerivativeVar(m.x3)
# @:vardecl


# @diffeq:
def _x1dot(m, t):
    return m.dx1[t] == m.x2[t]


m.x1dotcon = pyo.Constraint(m.t, rule=_x1dot)


def _x2dot(m, t):
    return m.dx2[t] == -m.x2[t] + m.u[t]


m.x2dotcon = pyo.Constraint(m.t, rule=_x2dot)


def _x3dot(m, t):
    return m.dx3[t] == m.x1[t] ** 2 + m.x2[t] ** 2 + 0.005 * m.u[t] ** 2


m.x3dotcon = pyo.Constraint(m.t, rule=_x3dot)
# @:diffeq

# @objpath:
m.obj = pyo.Objective(expr=m.x3[m.tf])


def _con(m, t):
    return m.x2[t] - 8 * (t - 0.5) ** 2 + 0.5 <= 0


m.con = pyo.Constraint(m.t, rule=_con)
# @:objpath

# @deactivate:
m.x1dotcon[m.t.first()].deactivate()
m.x2dotcon[m.t.first()].deactivate()
m.x3dotcon[m.t.first()].deactivate()
# @:deactivate

# @initcon:
m.x1[0].fix(0)
m.x2[m.t.first()].fix(-1)
m.x3[m.t.first()].fix(0)
# @:initcon

m.pprint()

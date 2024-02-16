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
from pyomo.dae import *

model = AbstractModel()

model.Feed = Param(initialize=24.0 / 60.0)
model.x_Feed = Param(initialize=0.5)
model.D = Param(initialize=model.x_Feed * model.Feed)
model.vol = Param(initialize=1.6)
model.atray = Param(initialize=0.25)
model.acond = Param(initialize=0.5)
model.areb = Param(initialize=1.0)

model.S_TRAYS = Set(dimen=1)
model.S_RECTIFICATION = Set(within=model.S_TRAYS)
model.S_STRIPPING = Set(within=model.S_TRAYS)
model.x0 = Param(model.S_TRAYS)

model.t = ContinuousSet(initialize=range(1, 52))
# Alternatively you could simply specify bounds on the
# ContinuousSet and let the finite element points be generated
# automatically.
# model.t = ContinuousSet(bounds=(1,51))

model.y = Var(model.S_TRAYS, model.t)


def x_init_rule(m, n, ti):
    return value(m.x0[n])


model.x = Var(model.S_TRAYS, model.t, initialize=x_init_rule)
model.dx = DerivativeVar(model.x)

model.rr = Var(model.t, initialize=3.0)
model.L = Var(model.t, initialize=0.6)
model.V = Var(model.t, initialize=0.8)
model.FL = Var(model.t, initialize=1)
model.u1 = Var(model.t, initialize=3.0, bounds=(1, 5))

model.alpha = Param(initialize=1000)
model.rho = Param(initialize=1)
model.u1_ref = Param(initialize=2.0)
model.y1_ref = Param(initialize=0.895814)


###
# Model constraints
###
def reflux_ratio_rule(m, t):
    return m.rr[t] == m.u1[t]


model.reflux_ratio = Constraint(model.t, rule=reflux_ratio_rule)


def flowrate_rectificaiton_rule(m, t):
    return m.L[t] == m.rr[t] * m.D


model.flowrate_rectificaiton = Constraint(model.t, rule=flowrate_rectificaiton_rule)


def vapor_column_rule(m, t):
    return m.V[t] == m.L[t] + m.D


model.vapor_column = Constraint(model.t, rule=vapor_column_rule)


def flowrate_stripping_rule(m, t):
    return m.FL[t] == m.Feed + m.L[t]


model.flowrate_stripping = Constraint(model.t, rule=flowrate_stripping_rule)


def mole_frac_balance_rule(m, n, t):
    return m.y[n, t] == m.x[n, t] * m.vol / (1 + ((m.vol - 1) * m.x[n, t]))


model.mole_frac_balance = Constraint(
    model.S_TRAYS, model.t, rule=mole_frac_balance_rule
)


def _diffeq(m, n, t):
    if t == 1:
        return Constraint.Skip
    if n == 1:
        return m.dx[n, t] == 1 / m.acond * m.V[t] * (m.y[n + 1, t] - m.x[n, t])
    elif n in m.S_RECTIFICATION:
        return m.dx[n, t] == 1 / m.atray * (
            m.L[t] * (m.x[n - 1, t] - m.x[n, t]) - m.V[t] * (m.y[n, t] - m.y[n + 1, t])
        )
    elif n == 17:
        return m.dx[n, t] == 1 / m.atray * (
            m.Feed * m.x_Feed
            + m.L[t] * m.x[n - 1, t]
            - m.FL[t] * m.x[n, t]
            - m.V[t] * (m.y[n, t] - m.y[n + 1, t])
        )
    elif n in m.S_STRIPPING:
        return m.dx[n, t] == 1 / m.atray * (
            m.FL[t] * (m.x[n - 1, t] - m.x[n, t]) - m.V[t] * (m.y[n, t] - m.y[n + 1, t])
        )
    else:
        return m.dx[n, t] == 1 / m.areb * (
            m.FL[t] * m.x[n - 1, t] - (m.Feed - m.D) * m.x[n, t] - m.V[t] * m.y[n, t]
        )


model.diffeq = Constraint(model.S_TRAYS, model.t, rule=_diffeq)


def _init_rule(m, n):
    return m.x[n, 1] == m.x0[n]


model.init_rule = Constraint(model.S_TRAYS, rule=_init_rule)

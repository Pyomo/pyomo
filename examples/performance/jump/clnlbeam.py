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

model = AbstractModel()

model.N = Param(within=PositiveIntegers)
model.h = 1.0 / model.N

model.VarIdx = RangeSet(model.N + 1)

model.t = Var(
    model.VarIdx, bounds=(-1.0, 1.0), initialize=lambda m, i: 0.05 * cos(i * m.h)
)
model.x = Var(
    model.VarIdx, bounds=(-0.05, 0.05), initialize=lambda m, i: 0.05 * cos(i * m.h)
)
model.u = Var(model.VarIdx, initialize=0.01)

alpha = 350


def c_rule(m):
    ex = 0
    for i in m.VarIdx:
        if i == m.N + 1:
            continue
        ex += 0.5 * m.h * (m.u[i + 1] ** 2 + m.u[i] ** 2) + 0.5 * alpha * m.h * (
            cos(m.t[i + 1]) + cos(m.t[i])
        )
    return ex


model.c = Objective(rule=c_rule)


def cons1_rule(m, i):
    if i == m.N + 1:
        return Constraint.Skip
    return m.x[i + 1] - m.x[i] - (0.5 * m.h) * (sin(m.t[i + 1]) + sin(m.t[i])) == 0


model.cons1 = Constraint(model.VarIdx, rule=cons1_rule)


def cons2_rule(m, i):
    if i == m.N + 1:
        return Constraint.Skip
    return m.t[i + 1] - m.t[i] - (0.5 * m.h) * m.u[i + 1] - (0.5 * m.h) * m.u[i] == 0


model.cons2 = Constraint(model.VarIdx, rule=cons2_rule)

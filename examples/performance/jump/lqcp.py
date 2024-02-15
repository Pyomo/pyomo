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

from pyomo.core import *

model = ConcreteModel()

model.n = 1000
model.m = 1000
model.dx = 1.0 / model.n
model.T = 1.58
model.dt = model.T / model.n
model.h2 = model.dx**2
model.a = 0.001

model.ns = RangeSet(0, model.n)
model.ms = RangeSet(0, model.n)

model.y = Var(model.ms, model.ns, bounds=(0.0, 1.0))
model.u = Var(model.ms, bounds=(-1.0, 1.0))


def yt(j, dx):
    return 0.5 * (1 - (j * dx) * (j * dx))


def rule(model):
    return 0.25 * model.dx * (
        (model.y[model.m, 0] - yt(0, model.dx)) ** 2
        + 2
        * sum((model.y[model.m, j] - yt(j, model.dx)) ** 2 for j in range(1, model.n))
        + (model.y[model.m, model.n] - yt(model.n, model.dx)) ** 2
    ) + 0.25 * model.a * model.dt * (
        2 * sum(model.u[i] ** 2 for i in range(1, model.m)) + model.u[model.m] ** 2
    )


model.obj = Objective(rule=rule)


def pde_rule(model, i, j):
    return (model.y[i + 1, j] - model.y[i, j]) / model.dt == 0.5 * (
        model.y[i, j - 1]
        - 2 * model.y[i, j]
        + model.y[i, j + 1]
        + model.y[i + 1, j - 1]
        - 2 * model.y[i + 1, j]
        + model.y[i + 1, j + 1]
    ) / model.h2


model.pde = Constraint(
    RangeSet(0, model.n - 1), RangeSet(1, model.n - 1), rule=pde_rule
)


def ic_rule(model, j):
    return model.y[0, j] == 0


model.ic = Constraint(model.ns, rule=ic_rule)


def bc1_rule(model, i):
    return model.y[i, 2] - 4 * model.y[i, 1] + 3 * model.y[i, 0] == 0


model.bc1 = Constraint(RangeSet(1, model.n), rule=bc1_rule)


def bc2_rule(model, i):
    return model.y[i, model.n - 2] - 4 * model.y[i, model.n - 1] + 3 * model.y[
        i, model.n - 0
    ] == (2 * model.dx) * (model.u[i] - model.y[i, model.n - 0])


model.bc2 = Constraint(RangeSet(1, model.n), rule=bc2_rule)

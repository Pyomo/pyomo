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

model = pyo.AbstractModel()

model.G = 25  # pyo.Param(within=pyo.PositiveIntegers)
model.F = 25  # pyo.Param(within=pyo.PositiveIntegers)

model.Grid = pyo.RangeSet(0, model.G)
model.Facs = pyo.RangeSet(1, model.F)
model.Dims = pyo.RangeSet(1, 2)

model.d = pyo.Var()
model.y = pyo.Var(model.Facs, model.Dims, bounds=(0.0, 1.0))
model.z = pyo.Var(model.Grid, model.Grid, model.Facs, within=pyo.Binary)
model.s = pyo.Var(model.Grid, model.Grid, model.Facs, bounds=(0.0, None))
model.r = pyo.Var(model.Grid, model.Grid, model.Facs, model.Dims)


def obj_rule(mod):
    return 1.0 * mod.d


model.obj = pyo.Objective(rule=obj_rule)


def assmt_rule(mod, i, j):
    return sum([mod.z[i, j, f] for f in mod.Facs]) == 1


model.assmt = pyo.Constraint(model.Grid, model.Grid, rule=assmt_rule)

M = 2 * 1.414


def quadrhs_rule(mod, i, j, f):
    return mod.s[i, j, f] == mod.d + M * (1 - mod.z[i, j, f])


model.quadrhs = pyo.Constraint(model.Grid, model.Grid, model.Facs, rule=quadrhs_rule)


def quaddistk1_rule(mod, i, j, f):
    return mod.r[i, j, f, 1] == (1.0 * i) / mod.G - mod.y[f, 1]


model.quaddistk1 = pyo.Constraint(
    model.Grid, model.Grid, model.Facs, rule=quaddistk1_rule
)


def quaddistk2_rule(mod, i, j, f):
    return mod.r[i, j, f, 2] == (1.0 * j) / mod.G - mod.y[f, 2]


model.quaddistk2 = pyo.Constraint(
    model.Grid, model.Grid, model.Facs, rule=quaddistk2_rule
)


def quaddist_rule(mod, i, j, f):
    return mod.r[i, j, f, 1] ** 2 + mod.r[i, j, f, 2] ** 2 <= mod.s[i, j, f] ** 2


model.quaddist = pyo.Constraint(model.Grid, model.Grid, model.Facs, rule=quaddist_rule)

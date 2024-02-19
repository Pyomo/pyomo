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

# AbstractHLinear.py - A simple linear version of (H)
import pyomo.environ as pyo

model = pyo.AbstractModel(name="Simple Linear (H)")

model.A = pyo.Set()

model.h = pyo.Param(model.A)
model.d = pyo.Param(model.A)
model.c = pyo.Param(model.A)
model.b = pyo.Param()
model.u = pyo.Param(model.A)


def xbounds_rule(model, i):
    return (0, model.u[i])


model.x = pyo.Var(model.A, bounds=xbounds_rule)


# @obj:
def obj_rule(model):
    return sum(
        model.h[i] * (1 - model.u[i] / model.d[i] ** 2) * model.x[i] for i in model.A
    )


# @:obj

model.z = pyo.Objective(rule=obj_rule, sense=pyo.maximize)


def budget_rule(model):
    return pyo.summation(model.c, model.x) <= model.b


model.budgetconstr = pyo.Constraint(rule=budget_rule)

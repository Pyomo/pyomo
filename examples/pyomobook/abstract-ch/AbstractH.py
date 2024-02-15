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

# AbstractH.py - Implement model (H)
import pyomo.environ as pyo

model = pyo.AbstractModel(name="(H)")

model.A = pyo.Set()

model.h = pyo.Param(model.A)
model.d = pyo.Param(model.A)
model.c = pyo.Param(model.A)
model.b = pyo.Param()
model.u = pyo.Param(model.A)


def xbounds_rule(model, i):
    return (0, model.u[i])


model.x = pyo.Var(model.A, bounds=xbounds_rule)


def obj_rule(model):
    return sum(
        model.h[i] * (model.x[i] - (model.x[i] / model.d[i]) ** 2) for i in model.A
    )


model.z = pyo.Objective(rule=obj_rule, sense=pyo.maximize)


def budget_rule(model):
    return sum(model.c[i] * model.x[i] for i in model.A) <= model.b


model.budgetconstr = pyo.Constraint(rule=budget_rule)

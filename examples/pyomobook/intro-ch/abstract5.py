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

import pyomo.environ as pyo

model = pyo.AbstractModel()

model.N = pyo.Set()
model.M = pyo.Set()
model.c = pyo.Param(model.N)
model.a = pyo.Param(model.M, model.N)
model.b = pyo.Param(model.M)

model.x = pyo.Var(model.N, within=pyo.NonNegativeReals)


def obj_rule(model):
    return sum(model.c[i] * model.x[i] for i in model.N)


model.obj = pyo.Objective(rule=obj_rule)


def con_rule(model, m):
    return sum(model.a[m, i] * model.x[i] for i in model.N) >= model.b[m]


model.con = pyo.Constraint(model.M, rule=con_rule)

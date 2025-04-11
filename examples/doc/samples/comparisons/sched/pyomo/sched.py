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

model.TASKS = pyo.Set()
model.PEOPLE = pyo.Set()
model.SLOTS = pyo.Set()

model.amt = pyo.Param(model.TASKS, model.PEOPLE, within=pyo.NonNegativeReals)
model.nrooms = pyo.Param(model.SLOTS, within=pyo.NonNegativeReals)
model.ntasks = pyo.Param(model.TASKS, within=pyo.NonNegativeReals)
model.minp = pyo.Param(model.TASKS, within=pyo.NonNegativeReals)


def maxp_valid(value, i, Model):
    return Model.maxp[i] >= Model.minp[i]


model.maxp = pyo.Param(model.TASKS, validate=maxp_valid)

model.x = pyo.Var(model.TASKS, model.PEOPLE, model.SLOTS, within=pyo.Binary)
model.xts = pyo.Var(model.TASKS, model.SLOTS, within=pyo.Binary)
model.xtp = pyo.Var(model.TASKS, model.PEOPLE, within=pyo.Binary)


def rule1_rule(t, s, Model):
    return sum(Model.x[t, p, s] for p in Model.PEOPLE) >= Model.xts[t, s]


model.rule1 = pyo.Constraint(model.TASKS, model.SLOTS)


def rule2_rule(t, p, s, Model):
    return Model.x[t, p, s] <= Model.xts[t, s]


model.rule2 = pyo.Constraint(model.TASKS, model.PEOPLE, model.SLOTS)


def rule3_rule(t, p, Model):
    return sum(Model.x[t, p, s] for s in Model.SLOTS) == Model.xtp[t, p]


model.rule3 = pyo.Constraint(model.TASKS, model.PEOPLE)


def rule4_rule(t, Model):
    return sum(Model.xts[t, s] for s in Model.SLOTS) == Model.ntasks[t]


model.rule4 = pyo.Constraint(model.TASKS)


def rule5_rule(t, Model):
    return Model.minp[t] <= sum(Model.xtp[t, p] for p in Model.PEOPLE) <= Model.maxp[t]


model.rule5 = pyo.Constraint(model.TASKS)


def rule6_rule(s, Model):
    return sum(Model.xts[t, s] for t in Model.TASKS) <= Model.nrooms[s]


model.rule6 = pyo.Constraint(model.SLOTS)


def rule7_rule(p, s, Model):
    return sum(Model.x[t, p, s] for t in Model.TASKS) == 1


model.rule7 = pyo.Constraint(model.PEOPLE, model.SLOTS)


def z_rule(Model):
    return sum(
        Model.amt[t, p] * Model.xtp[t, p] for t in Model.TASKS for p in Model.PEOPLE
    )


model.z = pyo.Objective(sense=pyo.maximize)

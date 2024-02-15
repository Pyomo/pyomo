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

model = AbstractModel()

model.TASKS = Set()
model.PEOPLE = Set()
model.SLOTS = Set()

model.amt = Param(model.TASKS, model.PEOPLE, within=NonNegativeReals)
model.nrooms = Param(model.SLOTS, within=NonNegativeReals)
model.ntasks = Param(model.TASKS, within=NonNegativeReals)
model.minp = Param(model.TASKS, within=NonNegativeReals)


def maxp_valid(value, i, Model):
    return Model.maxp[i] >= Model.minp[i]


model.maxp = Param(model.TASKS, validate=maxp_valid)

model.x = Var(model.TASKS, model.PEOPLE, model.SLOTS, within=Binary)
model.xts = Var(model.TASKS, model.SLOTS, within=Binary)
model.xtp = Var(model.TASKS, model.PEOPLE, within=Binary)


def rule1_rule(t, s, Model):
    return sum(Model.x[t, p, s] for p in Model.PEOPLE) >= Model.xts[t, s]


model.rule1 = Constraint(model.TASKS, model.SLOTS)


def rule2_rule(t, p, s, Model):
    return Model.x[t, p, s] <= Model.xts[t, s]


model.rule2 = Constraint(model.TASKS, model.PEOPLE, model.SLOTS)


def rule3_rule(t, p, Model):
    return sum(Model.x[t, p, s] for s in Model.SLOTS) == Model.xtp[t, p]


model.rule3 = Constraint(model.TASKS, model.PEOPLE)


def rule4_rule(t, Model):
    return sum(Model.xts[t, s] for s in Model.SLOTS) == Model.ntasks[t]


model.rule4 = Constraint(model.TASKS)


def rule5_rule(t, Model):
    return Model.minp[t] <= sum(Model.xtp[t, p] for p in Model.PEOPLE) <= Model.maxp[t]


model.rule5 = Constraint(model.TASKS)


def rule6_rule(s, Model):
    return sum(Model.xts[t, s] for t in Model.TASKS) <= Model.nrooms[s]


model.rule6 = Constraint(model.SLOTS)


def rule7_rule(p, s, Model):
    return sum(Model.x[t, p, s] for t in Model.TASKS) == 1


model.rule7 = Constraint(model.PEOPLE, model.SLOTS)


def z_rule(Model):
    return sum(
        Model.amt[t, p] * Model.xtp[t, p] for t in Model.TASKS for p in Model.PEOPLE
    )


model.z = Objective(sense=maximize)

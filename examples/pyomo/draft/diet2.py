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

## variant of diet.py that presents constraints and objectives twice,
## to see how terms are collected

#
# Imports
#
from pyomo.core import *

#
# Setup
#

model = AbstractModel()
model.NUTR = Set()
model.FOOD = Set()

model.cost = Param(model.FOOD, within=NonNegativeReals)

model.f_min = Param(model.FOOD, within=NonNegativeReals)


def f_max_valid(model, value, j):
    return model.f_max[j] > model.f_min[j]


model.f_max = Param(model.FOOD, validate=f_max_valid)

model.n_min = Param(model.NUTR, within=NonNegativeReals)


def paramn_max(model, i):
    model.n_max[i] > model.n_min[i]
    return model.n_max[i]


model.n_max = Param(model.NUTR, initialize=paramn_max)

# ***********************************

model.amt = Param(model.NUTR, model.FOOD, within=NonNegativeReals)


def Buy_bounds(model, i):
    return (model.f_min[i], model.f_max[i])


model.Buy = Var(model.FOOD, bounds=Buy_bounds)


def Objective_rule(model):
    ans = 0
    for j in model.FOOD:
        ans = ans + model.cost[j] * model.Buy[j]
    for j in model.FOOD:
        ans = ans + model.cost[j] * model.Buy[j]
    return ans


model.totalcost = Objective(rule=Objective_rule)


def Diet_rule(model, i):
    expr = 0
    for j in model.FOOD:
        expr = expr + model.amt[i, j] * model.Buy[j]
    for j in model.FOOD:
        expr = expr + model.amt[i, j] * model.Buy[j]
    expr = expr > 2 * model.n_min[i]
    expr = expr < 2 * model.n_max[i]
    return expr


model.Diet = Constraint(model.NUTR, rule=Diet_rule)

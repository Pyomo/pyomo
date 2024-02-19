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

#
# Imports
#
from pyomo.environ import *

infinity = float('inf')

#
# Model
#

model = AbstractModel()

model.NUTR = Set()
model.FOOD = Set()

model.cost = Param(model.FOOD, within=PositiveReals)

model.f_min = Param(model.FOOD, within=NonNegativeReals, default=0.0)


def f_max_validate(model, value, j):
    return model.f_max[j] > model.f_min[j]


model.f_max = Param(model.FOOD, validate=f_max_validate, default=infinity)

model.n_min = Param(model.NUTR, within=NonNegativeReals, default=0.0)


def n_max_validate(model, value, j):
    return value > model.n_min[j]


model.n_max = Param(model.NUTR, validate=n_max_validate, default=infinity)

model.amt = Param(model.NUTR, model.FOOD, within=NonNegativeReals)

# --------------------------------------------------------


def Buy_bounds(model, i):
    return (model.f_min[i], model.f_max[i])


model.Buy = Var(model.FOOD, bounds=Buy_bounds, within=NonNegativeIntegers)

# --------------------------------------------------------


def Total_Cost_rule(model):
    ans = 0
    for j in model.FOOD:
        ans = ans + model.cost[j] * model.Buy[j]
    return ans


model.Total_Cost = Objective(rule=Total_Cost_rule, sense=minimize)


def Nutr_Amt_rule(model, i):
    ans = 0
    for j in model.FOOD:
        ans = ans + model.amt[i, j] * model.Buy[j]
    return ans


# model.Nutr_Amt = Objective(model.NUTR, rule=Nutr_Amt_rule)

# --------------------------------------------------------


def Diet_rule(model, i):
    expr = 0
    for j in model.FOOD:
        expr = expr + model.amt[i, j] * model.Buy[j]
    return (model.n_min[i], expr, model.n_max[i])


model.Diet = Constraint(model.NUTR, rule=Diet_rule)

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

#
# Imports
#
import pyomo.environ as pyo

infinity = float('inf')

#
# Model
#

model = pyo.AbstractModel()

model.NUTR = pyo.Set()
model.FOOD = pyo.Set()

model.cost = pyo.Param(model.FOOD, within=pyo.PositiveReals)

model.f_min = pyo.Param(model.FOOD, within=pyo.NonNegativeReals, default=0.0)


def f_max_validate(model, value, j):
    return model.f_max[j] > model.f_min[j]


model.f_max = pyo.Param(model.FOOD, validate=f_max_validate, default=infinity)

model.n_min = pyo.Param(model.NUTR, within=pyo.NonNegativeReals, default=0.0)


def n_max_validate(model, value, j):
    return value > model.n_min[j]


model.n_max = pyo.Param(model.NUTR, validate=n_max_validate, default=infinity)

model.amt = pyo.Param(model.NUTR, model.FOOD, within=pyo.NonNegativeReals)

# --------------------------------------------------------


def Buy_bounds(model, i):
    return (model.f_min[i], model.f_max[i])


model.Buy = pyo.Var(model.FOOD, bounds=Buy_bounds, within=pyo.NonNegativeIntegers)

# --------------------------------------------------------


def Total_Cost_rule(model):
    ans = 0
    for j in model.FOOD:
        ans = ans + model.cost[j] * model.Buy[j]
    return ans


model.Total_Cost = pyo.Objective(rule=Total_Cost_rule, sense=pyo.minimize)


def Nutr_Amt_rule(model, i):
    ans = 0
    for j in model.FOOD:
        ans = ans + model.amt[i, j] * model.Buy[j]
    return ans


# model.Nutr_Amt = pyo.Objective(model.NUTR, rule=Nutr_Amt_rule)

# --------------------------------------------------------


def Diet_rule(model, i):
    expr = 0
    for j in model.FOOD:
        expr = expr + model.amt[i, j] * model.Buy[j]
    return (model.n_min[i], expr, model.n_max[i])


model.Diet = pyo.Constraint(model.NUTR, rule=Diet_rule)

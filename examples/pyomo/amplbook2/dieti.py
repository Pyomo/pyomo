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

#
# Setup
#

model = pyo.AbstractModel()

model.NUTR = pyo.Set()

model.FOOD = pyo.Set()

model.cost = pyo.Param(model.FOOD, within=pyo.NonNegativeReals)

model.f_min = pyo.Param(model.FOOD, within=pyo.NonNegativeReals)


def f_max_valid(model, value, j):
    return value > model.f_min[j]


model.f_max = pyo.Param(model.FOOD, validate=f_max_valid)

model.n_min = pyo.Param(model.NUTR, within=pyo.NonNegativeReals)


def paramn_max(model, value, i):
    return value > model.n_min[i]


model.n_max = pyo.Param(model.NUTR, validate=paramn_max)

model.amt = pyo.Param(model.NUTR, model.FOOD, within=pyo.NonNegativeReals)


def Buy_bounds(model, i):
    return (model.f_min[i], model.f_max[i])


model.Buy = pyo.Var(model.FOOD, bounds=Buy_bounds, domain=pyo.Integers)


def Objective_rule(model):
    return pyo.sum_product(model.cost, model.Buy)


model.totalcost = pyo.Objective(rule=Objective_rule)


def Diet_rule(model, i):
    expr = 0
    for j in model.FOOD:
        expr = expr + model.amt[i, j] * model.Buy[j]
    return (model.n_min[i], expr, model.n_max[i])


model.Diet = pyo.Constraint(model.NUTR, rule=Diet_rule)

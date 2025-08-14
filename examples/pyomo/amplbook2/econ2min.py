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

# ***********************************

model.PROD = pyo.Set(doc='products')

model.ACT = pyo.Set(doc='activities')

# ***********************************

model.cost = pyo.Param(
    model.ACT, within=pyo.PositiveReals, doc='cost per unit of each activity'
)

model.demand = pyo.Param(
    model.PROD, within=pyo.NonNegativeReals, doc='units of demand for each product'
)

model.io = pyo.Param(
    model.PROD,
    model.ACT,
    within=pyo.NonNegativeReals,
    doc='units of each product from 1 unit of each activity',
)

model.level_min = pyo.Param(
    model.ACT, within=pyo.NonNegativeReals, doc='min allowed level for each activity'
)

model.level_max = pyo.Param(
    model.ACT, within=pyo.NonNegativeReals, doc='max allowed level for each activity'
)

# ***********************************


def Level_bounds(model, i):
    return (model.level_min[i], model.level_max[i])


model.Level = pyo.Var(model.ACT, bounds=Level_bounds, doc='level for each activity')

# ***********************************


def Total_Cost_rule(model):
    return pyo.sum_product(model.cost, model.Level)


model.Total_Cost = pyo.Objective(rule=Total_Cost_rule, doc='minimize total cost')


def Demand_rule(model, i):
    expr = 0
    for j in model.ACT:
        expr += model.io[i, j] * model.Level[j]
    return model.demand[i] < expr


model.Demand = pyo.Constraint(
    model.PROD, rule=Demand_rule, doc='total level for each activity exceeds demand'
)

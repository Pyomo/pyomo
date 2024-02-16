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
from pyomo.core import *

#
# Setup
#

model = AbstractModel()

# ***********************************

model.PROD = Set(doc='products')

model.ACT = Set(doc='activities')

# ***********************************

model.cost = Param(
    model.ACT, within=PositiveReals, doc='cost per unit of each activity'
)

model.demand = Param(
    model.PROD, within=NonNegativeReals, doc='units of demand for each product'
)

model.io = Param(
    model.PROD,
    model.ACT,
    within=NonNegativeReals,
    doc='units of each product from 1 unit of each activity',
)

model.level_min = Param(
    model.ACT, within=NonNegativeReals, doc='min allowed level for each activity'
)

model.level_max = Param(
    model.ACT, within=NonNegativeReals, doc='max allowed level for each activity'
)

# ***********************************


def Level_bounds(model, i):
    return (model.level_min[i], model.level_max[i])


model.Level = Var(model.ACT, bounds=Level_bounds, doc='level for each activity')

# ***********************************


def Total_Cost_rule(model):
    return sum_product(model.cost, model.Level)


model.Total_Cost = Objective(rule=Total_Cost_rule, doc='minimize total cost')


def Demand_rule(model, i):
    expr = 0
    for j in model.ACT:
        expr += model.io[i, j] * model.Level[j]
    return model.demand[i] < expr


model.Demand = Constraint(
    model.PROD, rule=Demand_rule, doc='total level for each activity exceeds demand'
)

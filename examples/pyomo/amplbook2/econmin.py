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

model.PROD = pyo.Set()

model.ACT = pyo.Set()

# ***********************************

model.cost = pyo.Param(model.ACT, within=pyo.PositiveReals)

model.demand = pyo.Param(model.PROD, within=pyo.NonNegativeReals)

model.io = pyo.Param(model.PROD, model.ACT, within=pyo.NonNegativeReals)

# ***********************************

model.Level = pyo.Var(model.ACT)

# ***********************************


def Total_Cost_rule(model):
    return pyo.sum_product(model.cost, model.Level)


model.Total_Cost = pyo.Objective(rule=Total_Cost_rule)


def Demand_rule(model, i):
    expr = 0
    for j in model.ACT:
        expr += model.io[i, j] * model.Level[j]
    return expr > model.demand[i]


model.Demand = pyo.Constraint(model.PROD, rule=Demand_rule)

#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Imports
#
from pyomo.core import *

#
# Setup
#

model = AbstractModel()

# ***********************************

model.PROD = Set()

model.ACT = Set()

# ***********************************

model.cost = Param(model.ACT, within=PositiveReals)

model.demand = Param(model.PROD, within=NonNegativeReals)

model.io = Param(model.PROD, model.ACT, within=NonNegativeReals)

# ***********************************

model.Level = Var(model.ACT)

# ***********************************

def Total_Cost_rule(model):
    return summation(model.cost, model.Level)
model.Total_Cost = Objective(rule=Total_Cost_rule)

def Demand_rule(model, i):
    expr = 0
    for j in model.ACT:
        expr += model.io[i,j] * model.Level[j]
    return expr > model.demand[i]
model.Demand = Constraint(model.PROD, rule=Demand_rule)

#
# Imports
#
from coopr.pyomo import *

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

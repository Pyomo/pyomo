#
# Imports
#
from pyomo.core import *

#
# Setup
#

model = AbstractModel()

model.PROD = Set()

model.rate = Param(model.PROD, within=PositiveReals)

model.avail = Param(within=NonNegativeReals)

model.profit = Param(model.PROD)

model.commit = Param(model.PROD, within=NonNegativeReals)

model.market = Param(model.PROD, within=NonNegativeReals)

def Make_bounds(model, i):
    return (model.commit[i],model.market[i])
model.Make = Var(model.PROD, bounds=Make_bounds)

def Objective_rule(model):
    return summation(model.profit, model.Make)
model.totalprofit = Objective(rule=Objective_rule, sense=maximize)

def Time_rule(model):
    return summation(model.Make, denom=(model.rate)) < model.avail
model.Time = Constraint(rule=Time_rule)

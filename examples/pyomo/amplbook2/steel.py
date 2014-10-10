#
# Imports
#
from pyomo.core import *

#
# Setup
#

model = AbstractModel()

model.PROD = Set()

model.rate = Param(model.PROD,within=PositiveReals)

model.avail = Param(within=NonNegativeReals)

model.profit = Param(model.PROD)

model.market = Param(model.PROD, within=NonNegativeReals)

def Make_bounds(model,i):
    return (0,model.market[i])
model.Make = Var(model.PROD, bounds=Make_bounds)

def Objective_rule(model):
    return summation(model.profit, model.Make)
model.Total_Profit = Objective(rule=Objective_rule, sense=maximize)

def Time_rule(model):
    ans = 0
    for p in model.PROD:
        ans = ans + (1.0/model.rate[p]) * model.Make[p]
    return ans < model.avail

def XTime_rule(model):
    return summation(model.Make, denom=(model.rate,) ) < model.avail
#model.Time = Constraint(rule=Time_rule)

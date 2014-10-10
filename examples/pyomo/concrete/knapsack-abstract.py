#
# Abstract Knapsack Problem
#

from coopr.pyomo import *

model = AbstractModel()

model.ITEMS = Set()

model.v = Param(model.ITEMS, within=PositiveReals)

model.w = Param(model.ITEMS, within=PositiveReals)

model.limit = Param(within=PositiveReals)

model.x = Var(model.ITEMS, within=Binary)

def value_rule(model):
    return sum(model.v[i]*model.x[i] for i in model.ITEMS)
model.value = Objective(sense=maximize)

def weight_rule(model):
    return sum(model.w[i]*model.x[i] for i in model.ITEMS) <= model.limit
model.weight = Constraint()

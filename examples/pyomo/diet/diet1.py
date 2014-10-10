#
# Imports
#
from coopr.pyomo import *

infinity = float('inf')

#
# Model
#

model = AbstractModel()

model.FOOD = Set()

model.cost = Param(model.FOOD, within=PositiveReals)

model.f_min = Param(model.FOOD, within=NonNegativeReals, default=0.0)

MAX_FOOD_SUPPLY = 20.0 # McDonald's doesn't stock infinite food
def f_max_validate (model, value, j):
    return model.f_max[j] > model.f_min[j]
model.f_max = Param(model.FOOD, validate=f_max_validate, default=MAX_FOOD_SUPPLY)

# Unneeded vars - they're in the .dat file, so we list them here
model.NUTR = Set()
model.n_min = Param(model.NUTR, within=NonNegativeReals, default=0.0)
model.n_max = Param(model.NUTR, default=infinity)
model.amt = Param(model.NUTR, model.FOOD, within=NonNegativeReals)

# --------------------------------------------------------

def Buy_bounds(model, i):
    return (model.f_min[i], model.f_max[i])
model.Buy = Var(model.FOOD, bounds=Buy_bounds, within=NonNegativeIntegers)

# --------------------------------------------------------

def Total_Cost_rule(model):
    return sum(model.cost[j] * model.Buy[j] for j in model.FOOD)
model.Total_Cost = Objective(rule=Total_Cost_rule, sense=minimize)

# --------------------------------------------------------

def Entree_rule(model):
    entrees = ['Quarter Pounder w Cheese', 'McLean Deluxe w Cheese', 'Big Mac', 'Filet-O-Fish', 'McGrilled Chicken']
    return sum(model.Buy[e] for e in entrees) >= 1
model.Entree = Constraint(rule=Entree_rule)

def Side_rule(model):
    sides = ['Fries, small', 'Sausage McMuffin']
    return sum(model.Buy[s] for s in sides) >= 1
model.Side = Constraint(rule=Side_rule)

def Drink_rule(model):
    drinks = ['1% Lowfat Milk', 'Orange Juice']
    return sum(model.Buy[d] for d in drinks) >= 1
model.Drink = Constraint(rule=Drink_rule)

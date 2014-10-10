# Imports
from coopr.pyomo import *

# Create the model object
model = AbstractModel()

# Sets
model.P = Set()

# Parameters
model.a = Param(model.P)
model.b = Param()
model.c = Param(model.P)
model.u = Param(model.P)

# Variables
model.X = Var(model.P)

# Objective
def Objective_rule(model):
    return sum([model.c[j]*model.X[j] for j in model.P])
model.Total_Profit = Objective(rule=Objective_rule, sense=maximize)

# Time Constraint
def Time_rule(model):
    return summation(model.X, denom=model.a) <= model.b
model.Time = Constraint(rule=Time_rule)

# Limit Constraint
def Limit_rule(model, j):
    return (0, model.X[j], model.u[j])
model.Limit = Constraint(model.P, rule=Limit_rule)

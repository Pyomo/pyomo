# a simple quadratic function of two variables - taken from the CPLEX file format reference manual.
# optimal objective function value is 60. solution is x=10, y=0.

from coopr.pyomo import *

model = AbstractModel()

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)

def constraint_rule(model):
    return model.x + model.y >= 10
model.constraint = Constraint(rule=constraint_rule)

def objective_rule(model):
    return model.x + model.y + 0.5 * (model.x * model.x + 4 * model.x * model.y + 7 * model.y * model.y)
model.objective = Objective(rule=objective_rule, sense=minimize)

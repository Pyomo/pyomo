# a slightly less brain-dead separable parabolic function of three variables.
# optimal objective function value is obviously 5, with the optimal solution
# being x[1]=x[2]=x[3]=3.

from coopr.pyomo import *

model = AbstractModel()

def indices_rule(model):
    return xrange(1,4)
model.indices = Set(initialize=indices_rule, within=PositiveIntegers)

model.x = Var(model.indices, within=Reals)

def bound_x_rule(model, i):
    return (-10, model.x[i], 10)
model.bound_x = Constraint(model.indices, rule=bound_x_rule)

def objective_rule(model):
    return 5 + sum([(model.x[i] - 3) * (model.x[i] - 3) for i in model.indices])
model.objective = Objective(rule=objective_rule, sense=minimize)

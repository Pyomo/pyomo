from pyomo.core import *

model = AbstractModel()

model.indices = Set(initialize=[1,2])

model.p = Param(model.indices)

model.x = Var(model.indices)

def objective_rule ( M ):
    return sum([M.p[i] * M.x[i] for i in model.indices])

model.objective = Objective(rule=objective_rule, sense=minimize)

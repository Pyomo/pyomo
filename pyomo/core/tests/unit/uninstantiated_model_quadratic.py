from pyomo.environ import *

model = AbstractModel()

model.x = Var()

def objective_rule ( M ):
    return M.x * M.x    # should fail "gracefully"

model.objective = Objective(rule=objective_rule, sense=minimize)

# a brain-dead parabolic function of a single variable, whose minimum is
# obviously at x=0. the entry-level quadratic test-case. the lack of
# constraints could (but shouldn't in a perfect world) cause issues for
# certain solvers.

from pyomo.core import *

model = AbstractModel()

model.x = Var(bounds=(-10,10), within=Reals)

def objective_rule(model):
    return model.x * model.x
model.objective = Objective(rule=objective_rule, sense=minimize)

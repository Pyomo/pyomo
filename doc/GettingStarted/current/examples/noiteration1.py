# noiteration1.py

from pyomo.environ import *
from pyomo.opt import SolverFactory

# Create a solver
opt = SolverFactory('glpk')

#
# A simple model with binary variables and
# an empty constraint list.
#
model = AbstractModel()
model.n = Param(default=4)
model.x = Var(RangeSet(model.n), within=Binary)
def o_rule(model):
    return sum_product(model.x)
model.o = Objective(rule=o_rule)
model.c = ConstraintList()

# Create a model instance and optimize
instance = model.create_instance()
results = opt.solve(instance)
instance.solutions.load_from(results)

if instance.x[2].value == 0:
    print("The second index has a zero")
else:
    print("x[2]=",instance.x[2].value)


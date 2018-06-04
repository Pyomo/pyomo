# iterative1.py

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
instance.display()

# Iterate to eliminate the previously found solution
for i in range(5):
    instance.solutions.load_from(results)

    expr = 0
    for j in instance.x:
        if instance.x[j].value == 0:
            expr += instance.x[j]
        else:
            expr += (1-instance.x[j])
    instance.c.add( expr >= 1 )

    results = opt.solve(instance)
    print ("\n===== iteration",i)
    instance.display()

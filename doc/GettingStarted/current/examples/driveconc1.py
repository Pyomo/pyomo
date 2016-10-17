# driveconc1.py
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory

# Create a solver
opt = SolverFactory('cplex')

# get the model from another file
from concrete1 import model

# Create a 'dual' suffix component on the instance
# so the solver plugin will know which suffixes to collect
model.dual = Suffix(direction=Suffix.IMPORT)

results = opt.solve(model) # also load results to model

# display all duals
print ("Duals")
from pyomo.core import Constraint
for c in model.component_objects(Constraint, active=True):
    print ("   Constraint",c)
    cobject = getattr(model, str(c))
    for index in cobject:
        print ("      ", index, model.dual[cobject[index]])




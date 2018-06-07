# driveconc1.py
from __future__ import division
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Create a solver
opt = SolverFactory('cplex')

# get the model from another file
from concrete1 import model

# Create a 'dual' suffix component on the instance
# so the solver plugin will know which suffixes to collect
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

results = opt.solve(model) # also load results to model

# display all duals
print ("Duals")
for c in model.component_objects(pyo.Constraint, active=True):
    print ("   Constraint",c)
    for index in c:
        print ("      ", index, model.dual[c[index]])




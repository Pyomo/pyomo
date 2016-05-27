from pyomo.environ import *
from simple import model

#import logging
#logger = logging.getLogger('pyomo.opt')
#logger.setLevel(50)

# @load:
solver = SolverFactory('glpk')

# check final basis using exact arithmetic
#   --steep
solver.options.steep=None
# set glpk random number seed
#   --seed=1357908642
solver.options.seed=1357908642
# Run GLPK with the steep and seed solver options
results = solver.solve(model)

# Rerun GLPK with a new seed solver option
# This does NOT override the solver.options values
results = solver.solve(model, options={'seed':1234567890})
# @:load
print(solver.options.seed)

model.solutions.store_to(results)
print(results)

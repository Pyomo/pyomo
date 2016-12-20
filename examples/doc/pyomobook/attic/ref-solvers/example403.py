from pyomo.environ import *
from simple import model

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
# @:load

model.solutions.store_to(results)
print(results)
